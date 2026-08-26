import importlib.util, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gate', ROOT / 'tools' / 'validate_learn319_authority_gate.py')
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)

def test_gate_is_fail_closed_and_self_hashed():
    MOD.validate(ROOT / 'evidence' / 'learn319' / 'authority-gate.json')
