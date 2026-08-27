from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
VALIDATOR = ROOT / "tools/validate_next100_063_terminal_source_registry_v4.py"
EXPECTED_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_v4_registry_identity_and_validator_pass() -> None:
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
    assert "NEXT100-063 V4 PASS" in proc.stdout
    assert "numeric_capacity_bytes=2045180" in proc.stdout
    assert "en_numeric_bytes=1838293" in proc.stdout
    assert "families=14" in proc.stdout
    assert "gap_bytes=17954820" in proc.stdout


def test_v4_consumes_exact_artifacts_without_promoting_training() -> None:
    data = load_registry()
    rows = {row["pr"]: row for row in data["terminal_late_additions"]}

    cp = rows[467]
    assert cp["source_normalized_bytes"] == 17_901
    assert cp["numeric_training_capacity_bytes"] == 15_540
    assert cp["numeric_training_capacity_bytes"] < cp["source_normalized_bytes"]
    assert cp["accepted_chunk_count"] == 14
    assert cp["rejected_chunk_count"] == 2
    assert cp["accepted_materialization"]["workflow_run"] == 33_005_689_174
    assert cp["accepted_materialization"]["workflow_conclusion"] == "success"
    assert cp["accepted_materialization"]["artifact_id"] == 9_620_571_005
    assert cp["accepted_materialization"]["accepted_capacity_bytes"] == 15_540
    assert cp["accepted_materialization"]["rejection_reasons"] == {"pii_phone": 2}

    pg = rows[470]
    assert pg["source_record_count"] == 3
    assert pg["numeric_training_capacity_bytes"] == 1_672_110
    assert pg["family"] == "en.project-gutenberg.public-domain-books"
    assert pg["dedicated_workflow_run"] == 32_998_859_164
    assert pg["terminal_artifact_id"] == 9_618_402_768

    inv = data["pre_successor_global_dedup_inventory"]
    assert inv["candidate_numeric_training_capacity_bytes"] == 2_045_180
    assert inv["candidate_source_normalized_envelope_bytes"] == 2_047_541
    assert inv["uncredited_source_normalized_bytes"] == 2_361
    assert inv["candidate_independent_family_count"] == 14
    assert inv["by_stratum"]["en"]["numeric_training_capacity_bytes"] == 1_838_293
    assert inv["by_stratum"]["en"]["family_count"] == 5

    gates = data["downstream_gate_vector"]
    assert gates["authorized_balanced_no_replay_loss_positions"] == 0
    assert gates["successor_global_cross_source_exact_near_dedup"] == "REQUIRED_NEXT"
    assert gates["long_training"] == "BLOCKED"
    assert gates["paid_compute"] == "NOT_AUTHORIZED"


def test_v4_preserves_family_and_failure_boundaries() -> None:
    data = load_registry()
    parent_families = set(data["dedup_parent"]["families"])
    late = data["terminal_late_additions"]
    late_families = [row["family"] for row in late]

    assert len(late_families) == len(set(late_families)) == 7
    assert not (parent_families & set(late_families))
    assert len(parent_families | set(late_families)) == 14

    held = {row["pr"]: row for row in data["held_out_or_zero_credit"]}
    assert "COMPLETED_FAILURE" in held[465]["reason"]
    assert "COMPLETED_FAILURE" in held[475]["reason"]
    assert 470 not in held
