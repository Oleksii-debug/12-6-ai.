from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v3.json"
VALIDATOR = ROOT / "tools/validate_next100_063_terminal_source_registry_v3.py"
EXPECTED_IDENTITY = "66866a35d58b2f34431068a161986fc3eeb656e5ded1ca2ff8b40489049bac8c"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_v3_registry_identity_and_validator_pass() -> None:
    data = load_registry()
    claimed = data.pop("registry_identity_sha256")
    actual = hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    assert claimed == actual == EXPECTED_IDENTITY

    proc = subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "NEXT100-063 V3 PASS" in proc.stdout
    assert "numeric_capacity_bytes=357530" in proc.stdout
    assert "normalized_envelope_bytes=375431" in proc.stdout
    assert "families=13" in proc.stdout


def test_v3_has_no_double_credit_and_preserves_zero_credit_boundaries() -> None:
    data = load_registry()
    parent_families = set(data["dedup_parent"]["families"])
    late = {row["pr"]: row for row in data["terminal_late_additions"]}
    held = {row["pr"]: row for row in data["held_out_or_zero_credit"]}

    assert set(late) == {445, 449, 462, 467, 468, 472}
    assert not (parent_families & {row["family"] for row in late.values()})
    assert 468 not in held
    assert late[468]["numeric_training_capacity_bytes"] == 36_898
    assert late[468]["dedicated_workflow_conclusion"] == "success"

    assert late[467]["source_normalized_bytes"] == 17_901
    assert late[467]["numeric_training_capacity_bytes"] == 0
    assert late[467]["accepted_chunk_count"] == 14
    assert late[467]["rejected_chunk_count"] == 2

    assert "COMPLETED_FAILURE" in held[465]["reason"]
    assert "COMPLETED_FAILURE" in held[475]["reason"]
    assert data["downstream_gate_vector"]["authorized_balanced_no_replay_loss_positions"] == 0
    assert data["downstream_gate_vector"]["long_training"] == "BLOCKED"
    assert data["downstream_gate_vector"]["paid_compute"] == "NOT_AUTHORIZED"
