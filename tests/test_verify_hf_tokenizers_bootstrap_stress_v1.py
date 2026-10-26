import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_hf_tokenizers_bootstrap_stress_v1 as m

GOOD = "5020afd671a3885c1b738c8b4eafe7525f630546"


class TokenizersBootstrapStressTests(unittest.TestCase):
    def base(self):
        offline = {
            "attempted": True,
            "success": False,
            "stdout": "",
            "stderr": "DNS offline",
            "exit_code": 128,
        }
        env = {
            "python": "test",
            "executable": "/tmp/venv/bin/python",
            "prefix": "/tmp/venv",
            "base_prefix": "/usr",
            "platform": "test",
            "os_name": "posix",
            "arch": "x86_64",
            "cpu_count": 1,
            "gpu": {"present": False},
            "package_managers": {"pip": "pip 1", "uv": None, "poetry": None, "pdm": None, "conda": None},
            "git": "git 1",
            "git_path": "/usr/bin/git",
            "cache": {"pip_dir": "/tmp/cache", "pip_cache_probe_error": None, "pip_tokenizers_cache_list": "", "pip_dir_exists": False},
            "identity_sha256": "test",
        }
        with patch.object(m, "source_probe", return_value=offline), patch.object(m, "env_snapshot", return_value=env):
            return m.build(GOOD, False)

    def test_floating_ref_rejected(self):
        x = copy.deepcopy(self.base())
        x["upstream"]["tag"] = "main"
        self.assertFalse(m.validate(x)[0])

    def test_version_drift_rejected(self):
        x = copy.deepcopy(self.base())
        x["upstream"]["version"] = "0.23.0"
        self.assertFalse(m.validate(x)[0])

    def test_wheel_hash_drift_rejected(self):
        x = copy.deepcopy(self.base())
        x["artifact"]["sha256"] = "0" * 64
        self.assertFalse(m.validate(x)[0])

    def test_global_install_rejected(self):
        x = copy.deepcopy(self.base())
        x["execution"]["global_install_intent"] = True
        self.assertFalse(m.validate(x)[0])

    def test_fabricated_success_rejected(self):
        x = copy.deepcopy(self.base())
        x["execution"].update({"status": "INSTALLED_AND_EXECUTED", "isolated": True, "install_success": True, "runtime_success": True, "source_fetch_success": False})
        self.assertFalse(m.validate(x)[0])

    def test_canonical_base_rejected(self):
        x = copy.deepcopy(self.base())
        x["canonical_base"]["foreign_weights"] = True
        self.assertFalse(m.validate(x)[0])

    def test_real_environment_snapshot(self):
        x = m.env_snapshot()
        self.assertTrue(x["python"])
        self.assertIn("pip", x["package_managers"])

    def test_isolated_venv_and_freeze(self):
        x = m.isolated_probe(False)
        self.assertTrue(x["venv_created"])
        self.assertTrue(x["isolated"])
        self.assertGreaterEqual(len(x["pip_freeze"]), 1)

    def test_offline_never_promotes(self):
        x = self.base()
        self.assertIn(x["execution"]["status"], {"NOT_EXECUTED", "RETEST_RUNTIME_REQUIRED"})
        self.assertNotEqual(x["promotion"], "ADOPTABLE_COMPONENT")

    def test_evidence_identity_repeatable(self):
        a = self.base()
        b = self.base()
        self.assertEqual(a["identity_sha256"], b["identity_sha256"])


if __name__ == "__main__":
    unittest.main()
