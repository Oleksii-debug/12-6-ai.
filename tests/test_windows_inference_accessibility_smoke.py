from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "run_windows_inference_accessibility_smoke.py"


def test_portable_inference_accessibility_smoke(tmp_path: Path) -> None:
    report_path = tmp_path / "windows-smoke.json"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--source-sha",
            "a" * 40,
            "--bundle-sha256",
            "b" * 64,
            "--report",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_payload = json.loads(result.stdout)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert stdout_payload == report_payload
    assert report_payload["schema"] == "12-6.windows-inference-accessibility-smoke.v1"
    assert report_payload["passed"] is True
    assert report_payload["source_sha"] == "a" * 40
    assert report_payload["source_bundle_sha256"] == "b" * 64
    assert set(report_payload["checks"].values()) == {"PASS"}
    assert report_payload["truth_boundary"]["canonical_first_party_checkpoint_on_windows"] == (
        "NOT_TESTED"
    )
    assert report_payload["truth_boundary"]["nvda_live_session"] == "NOT_TESTED"
    assert report_payload["truth_boundary"]["promotion_authority"] is False


def test_accessibility_smoke_rejects_abbreviated_source_sha() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), "--source-sha", "abc123"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "source_sha must be a lowercase full 40-hex Git SHA" in result.stderr
