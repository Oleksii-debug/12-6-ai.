from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/check_learned_20m_launch_gate.py"

spec = importlib.util.spec_from_file_location("check_learned_20m_launch_gate", MODULE_PATH)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_exact_model341_initspec_is_required() -> None:
    evidence = {
        "launch_binding": {"initspec_identity": cli.MODEL341_INITSPEC_SHA256}
    }
    assert cli._validate_exact_initspec(evidence) == []


def test_plausible_but_wrong_initspec_fails_closed() -> None:
    evidence = {"launch_binding": {"initspec_identity": "0" * 64}}
    assert cli._validate_exact_initspec(evidence) == [
        "launch_binding.initspec_identity_mismatch"
    ]


def test_missing_launch_binding_fails_closed() -> None:
    assert cli._validate_exact_initspec({}) == [
        "launch_binding.initspec_identity_missing"
    ]
