from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
PACKET = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
EXPECTED_BLOB_SHA1 = "c50154db609d41eceb2ffc97912360df567bcc04"


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()


def test_checked_in_r01_campaign_retains_exact_merged_blob() -> None:
    assert _git_blob_sha1(CAMPAIGN.read_bytes()) == EXPECTED_BLOB_SHA1


def test_cli_rejects_campaign_file_drift_before_readiness(tmp_path: Path) -> None:
    mutated = tmp_path / "r01_campaign.json"
    mutated.write_bytes(CAMPAIGN.read_bytes() + b"\n")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/assess_r01_learned20m_launch_readiness.py"),
            str(PACKET),
            str(mutated),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["error"] == "r01_campaign_file_identity_mismatch"
    assert payload["expected_git_blob_sha1"] == EXPECTED_BLOB_SHA1
    assert payload["observed_git_blob_sha1"] != EXPECTED_BLOB_SHA1
