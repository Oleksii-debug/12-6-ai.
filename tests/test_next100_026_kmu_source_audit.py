import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Next100026KmuSourceAuditTest(unittest.TestCase):
    def test_validator_passes(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "tools/validate_next100_026_kmu_source_audit.py")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        report = json.loads(p.stdout)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["records"], 6)
        self.assertEqual(report["normalized_bytes"], 9153)


if __name__ == "__main__":
    unittest.main()
