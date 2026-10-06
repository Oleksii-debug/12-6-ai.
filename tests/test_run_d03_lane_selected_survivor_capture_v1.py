from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "run_d03_lane_selected_survivor_capture_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_d03_lane_selected_survivor_capture_v1",
    MODULE,
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw(
    record_id: str,
    source_id: str,
    family: str,
    payload: str,
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": source_id,
        "family": family,
        "modality": "text",
        "normalized_payload": payload,
    }


def _fixture() -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any],
]:
    rows = [
        _raw("a", "source-a", "family-a", "A"),
        _raw("b", "source-b", "family-b", "BB"),
        _raw("c", "source-c", "family-b", "CCC"),
    ]
    authority: list[dict[str, Any]] = []
    for row in rows:
        payload = row["normalized_payload"].encode("utf-8")
        authority.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "family": row["family"],
                "modality": row["modality"],
                "payload_bytes": len(payload),
                "payload_sha256": _sha(payload),
            }
        )
    plan = {
        "family_materialization_plan": [
            {
                "family": "family-b",
                "missing_record_ids": ["b", "c"],
                "missing_record_count": 2,
                "missing_payload_bytes": 5,
            }
        ]
    }
    composition = {
        "combined_inventory": {
            "records": authority,
        }
    }
    return rows, plan, composition


@pytest.fixture(autouse=True)
def _synthetic_plan_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RUNNER.assembly_api,
        "verify_plan",
        lambda _value: None,
    )


def test_authenticate_capture_exact_target_is_text_free() -> None:
    rows, plan, composition = _fixture()
    output, receipt = RUNNER.authenticate_capture(
        captured_rows=rows,
        plan=plan,
        composition=composition,
        families=frozenset({"family-b"}),
        lane_head="a" * 40,
        runner_blob="b" * 40,
        hook_path="clean.execute",
    )
    assert [row["record_id"] for row in output] == ["b", "c"]
    assert receipt["selected_record_count"] == 2
    assert receipt["selected_payload_bytes"] == 5
    assert receipt["raw_output_is_ephemeral"] is True
    assert receipt["durable_output_contains_raw_payload"] is False
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["training_executed"] is False
    assert b"normalized_payload" not in RUNNER.canonical(receipt)


def test_payload_tamper_is_rejected() -> None:
    rows, plan, composition = _fixture()
    rows[1] = dict(rows[1])
    rows[1]["normalized_payload"] = "CC"
    with pytest.raises(RUNNER.CaptureError, match="payload SHA drift"):
        RUNNER.authenticate_capture(
            captured_rows=rows,
            plan=plan,
            composition=composition,
            families=frozenset({"family-b"}),
            lane_head="a" * 40,
            runner_blob="b" * 40,
            hook_path="clean.execute",
        )


def test_missing_selected_survivor_is_rejected() -> None:
    rows, plan, composition = _fixture()
    with pytest.raises(
        RUNNER.CaptureError,
        match="absent from lane survivors",
    ):
        RUNNER.authenticate_capture(
            captured_rows=rows[:2],
            plan=plan,
            composition=composition,
            families=frozenset({"family-b"}),
            lane_head="a" * 40,
            runner_blob="b" * 40,
            hook_path="clean.execute",
        )


def test_required_family_must_exist_in_plan() -> None:
    rows, plan, composition = _fixture()
    with pytest.raises(
        RUNNER.CaptureError,
        match="required family absent",
    ):
        RUNNER.authenticate_capture(
            captured_rows=rows,
            plan=plan,
            composition=composition,
            families=frozenset({"family-c"}),
            lane_head="a" * 40,
            runner_blob="b" * 40,
            hook_path="clean.execute",
        )


def test_capture_rejects_duplicate_survivor_record() -> None:
    rows, plan, composition = _fixture()
    with pytest.raises(
        RUNNER.CaptureError,
        match="captured record replay",
    ):
        RUNNER.authenticate_capture(
            captured_rows=[*rows, rows[1]],
            plan=plan,
            composition=composition,
            families=frozenset({"family-b"}),
            lane_head="a" * 40,
            runner_blob="b" * 40,
            hook_path="clean.execute",
        )


def test_split_cli_requires_explicit_lane_separator() -> None:
    with pytest.raises(
        RUNNER.CaptureError,
        match="must follow --",
    ):
        RUNNER._split_cli(["--runner", "fake.py"])


def _init_fake_lane(
    root: Path,
    rows: list[dict[str, str]],
) -> tuple[Path, str, str]:
    tools = root / "tools"
    tools.mkdir(parents=True)
    runner = tools / "fake_lane.py"
    payload = json.dumps(rows, ensure_ascii=False)
    runner.write_text(
        (
            "import json\n"
            "class Clean:\n"
            "    def execute(self):\n"
            f"        rows = json.loads({payload!r})\n"
            "        return (None, None, None, None, None, None, rows, {})\n"
            "clean = Clean()\n"
            "def main():\n"
            "    clean.execute()\n"
            "    return 0\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "lane"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    blob = RUNNER.git_blob_sha1(runner.read_bytes())
    return runner, head, blob


def test_end_to_end_pinned_hook_writes_only_selected_target(
    tmp_path: Path,
) -> None:
    rows, plan, composition = _fixture()
    runner, head, blob = _init_fake_lane(tmp_path / "lane", rows)
    plan_json = tmp_path / "plan.json"
    composition_json = tmp_path / "composition.json"
    output_raw = tmp_path / "selected.jsonl"
    output_receipt = tmp_path / "receipt.json"
    plan_json.write_text(
        json.dumps(plan, ensure_ascii=False),
        encoding="utf-8",
    )
    composition_json.write_text(
        json.dumps(composition, ensure_ascii=False),
        encoding="utf-8",
    )

    receipt = RUNNER.execute(
        runner_path=runner,
        expected_lane_head=head,
        expected_runner_blob=blob,
        hook_path="clean.execute",
        survivor_index=6,
        families=frozenset({"family-b"}),
        plan_json=plan_json,
        composition_json=composition_json,
        output_raw_jsonl=output_raw,
        output_receipt=output_receipt,
        runner_args=["--unused"],
    )

    raw_rows = [
        json.loads(line)
        for line in output_raw.read_text(
            encoding="utf-8",
        ).splitlines()
    ]
    assert [row["record_id"] for row in raw_rows] == ["b", "c"]
    assert receipt["selected_record_count"] == 2
    assert "normalized_payload" not in output_receipt.read_text(
        encoding="utf-8"
    )


def test_runner_blob_drift_fails_before_import(tmp_path: Path) -> None:
    rows, _plan, _composition = _fixture()
    runner, head, _blob = _init_fake_lane(tmp_path / "lane", rows)
    with pytest.raises(
        RUNNER.CaptureError,
        match="runner Git blob drift",
    ):
        RUNNER._load_runner(
            runner,
            expected_head=head,
            expected_blob="0" * 40,
        )


def test_create_only_output_refuses_replacement(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"sentinel\n")
    with pytest.raises(
        RUNNER.CaptureError,
        match="output already exists",
    ):
        RUNNER._write_create_only(target, b"replacement\n")
    assert target.read_bytes() == b"sentinel\n"
