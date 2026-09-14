from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_eval303_selection_validation_composite.py"
MANIFEST = Path("configs/evaluation/eval303_selection_validation_composite_v1.json")
MEMBERSHIP = Path("data/evaluation/eval303/selection-validation/composite-membership.jsonl")
PROOF = Path("evidence/eval303/data300-exact-exclusion-proof-v1.json")


def _copy_authority(tmp_path: Path) -> None:
    for relative in (MANIFEST, MEMBERSHIP, PROOF):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _run_optimized(repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-O",
            str(SCRIPT),
            "verify",
            "--repo-root",
            str(repo_root),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_optimized_python_accepts_exact_immutable_authority() -> None:
    result = _run_optimized(ROOT)
    assert result.returncode == 0, result.stderr
    assert '"status": "PASS"' in result.stdout


def test_optimized_python_keeps_eval303_fail_closed(tmp_path: Path) -> None:
    _copy_authority(tmp_path)
    manifest_path = tmp_path / MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["usage_contract"]["may_train"] = True
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _run_optimized(tmp_path)
    assert result.returncode != 0
    assert "Eval303ValidationError" in result.stderr
