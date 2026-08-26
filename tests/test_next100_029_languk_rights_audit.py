from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_next100_029_languk_rights_audit.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("next100_029_validator", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_languk_rights_audit_fails_closed():
    module = _load_validator()
    payload = module.validate()
    terminal = payload["terminal_result"]
    assert terminal["training_source_admitted"] is False
    assert terminal["registry_change_authorized"] is False


def test_only_court_decisions_survives_as_retest():
    module = _load_validator()
    payload = module.validate()
    states = {item["candidate_id"]: item["verdict"] for item in payload["candidates"]}
    assert states["languk.court-decisions-uk.supreme-2024-5k"] == "RETEST_PRIVACY_AND_BYTE_MATERIALIZATION"
    assert states["languk.ubertext"].startswith("REJECT_")
    assert states["brown-uk.corpus"].startswith("REJECT_")
    assert states["languk.malyuk"].startswith("REJECT_")
