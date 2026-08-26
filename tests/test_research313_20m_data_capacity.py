from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_research313_20m_data_capacity.py"

spec = importlib.util.spec_from_file_location("research313_validator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_research313_evidence_is_fail_closed_and_self_consistent():
    module.validate()
