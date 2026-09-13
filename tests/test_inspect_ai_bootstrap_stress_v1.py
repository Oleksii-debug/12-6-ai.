from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/research/inspect_ai_bootstrap_stress_v1.json"
VALIDATOR = ROOT / "tools/validate_inspect_ai_bootstrap_stress_v1.py"


def test_manifest_is_exact_and_not_adopted():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["qualification_id"] == "INSPECT-AI-BOOTSTRAP-STRESS-V1"
    assert data["status"] == "RETEST_RUNTIME_REQUIRED"
    assert data["upstream"]["release"] == "0.3.260"
    assert data["upstream"]["commit"] == "3f294e61b823d6bad5fc16706fc5825ea980c8ee"
    assert data["project"]["canonical_base_contamination"] is False
    assert data["project"]["foreign_weights_used"] is False


def test_fail_closed_validator_passes_evidence():
    result = subprocess.run([sys.executable, str(VALIDATOR)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    assert '"validation": "PASS"' in result.stdout


def test_false_adoption_claim_is_rejected(tmp_path):
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["status"] = "ADOPTABLE_COMPONENT"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    code = "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); import validate_inspect_ai_bootstrap_stress_v1 as v; d=json.loads(Path(sys.argv[2]).read_text()); assert 'adoptable_without_real_runtime_evidence' in v.validate(d)"
    result = subprocess.run([sys.executable, "-c", code, str(VALIDATOR), str(bad)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout


def test_runtime_evidence_is_explicitly_unexecuted():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["runtime"]["execution"] == "NOT_EXECUTED"
    assert data["benchmark"]["execution"] == "NOT_EXECUTED"
    assert data["parity"]["execution"] == "NOT_EXECUTED"
