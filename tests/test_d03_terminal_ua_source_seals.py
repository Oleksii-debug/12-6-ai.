from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_d03_terminal_ua_source_seals",
    ROOT / "tools" / "validate_d03_terminal_ua_source_seals.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
validate = MODULE.validate

CFG = json.loads(
    (ROOT / "configs/data/d03_terminal_ua_source_seals_v1.json").read_text(
        encoding="utf-8"
    )
)


def reseal(obj):
    obj.pop("authority_identity_sha256", None)
    raw = (
        json.dumps(
            obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    obj["authority_identity_sha256"] = hashlib.sha256(raw).hexdigest()
    return obj


def test_valid():
    validate(copy.deepcopy(CFG))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_workflow_conclusion", "queued"),
        ("terminal_verdict", "RETEST"),
        ("family", "php.manual.documentation"),
        ("normalized_bytes", 30511),
        ("normalized_bundle_sha256", "0" * 64),
        ("evaluation", "ALLOWED"),
    ],
)
def test_source_tamper_fails(field, value):
    obj = copy.deepcopy(CFG)
    obj["sources"][1][field] = value
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


def test_training_credit_promotion_fails():
    obj = copy.deepcopy(CFG)
    obj["aggregate"]["training_authorized_bytes"] = 1
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


def test_model_training_promotion_fails():
    obj = copy.deepcopy(CFG)
    obj["current_main_truth_boundary"]["model_training_executed"] = True
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


def test_gate_removal_fails():
    obj = copy.deepcopy(CFG)
    obj["required_downstream_gates"] = obj["required_downstream_gates"][:-1]
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


def test_unsealed_tamper_fails():
    obj = copy.deepcopy(CFG)
    obj["aggregate"]["normalized_source_bytes"] += 1
    with pytest.raises(ValueError):
        validate(obj)
