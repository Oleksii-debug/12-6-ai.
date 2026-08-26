import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_next100_026_validator_passes():
    p=subprocess.run([sys.executable,str(ROOT/"tools/validate_next100_026_kmu_source_audit.py")],cwd=ROOT,text=True,capture_output=True,check=True)
    report=json.loads(p.stdout)
    assert report["status"]=="PASS"
    assert report["records"]==6
    assert report["normalized_bytes"]==9153
