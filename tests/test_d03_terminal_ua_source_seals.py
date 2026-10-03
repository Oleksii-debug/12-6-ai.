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
load_strict_json_object = MODULE.load_strict_json_object

CFG_RAW = (ROOT / "configs/data/d03_terminal_ua_source_seals_v1.json").read_text(
    encoding="utf-8"
)
CFG = load_strict_json_object(CFG_RAW)


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
        ("repository", "https://example.invalid/forbidden"),
        ("modality", "unknown_modality"),
        ("license_ids", ["MIT"]),
        ("license_sha256", ["0" * 64]),
        ("redistribution", "UNRESTRICTED"),
        ("historical_gates_all_passed", 1),
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



@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "unexpected_training_promotion", True),
        ("aggregate", "unexpected_training_promotion", 1),
        ("current_main_truth_boundary", "unexpected_training_promotion", True),
    ],
)
def test_unknown_resealed_fields_fail_closed(section, field, value):
    obj = copy.deepcopy(CFG)
    target = obj if section is None else obj[section]
    target[field] = value
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("local_free_only",), 1),
        (("aggregate", "canonical_capacity_credit_bytes"), False),
        (("current_main_truth_boundary", "model_training_executed"), 0),
        (("sources", 0, "historical_gates_all_passed"), 1),
    ],
)
def test_bool_int_aliases_fail_closed(path, value):
    obj = copy.deepcopy(CFG)
    target = obj
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    reseal(obj)
    with pytest.raises(ValueError):
        validate(obj)


@pytest.mark.parametrize(
    "raw",
    [
        CFG_RAW.replace(
            '  "worker_id": "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN",',
            '  "worker_id": "CONFLICT",\n'
            '  "worker_id": "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN",',
            1,
        ),
        CFG_RAW.replace(
            '      "repository": "https://github.com/php/doc-uk",',
            '      "repository": "https://example.invalid/conflict",\n'
            '      "repository": "https://github.com/php/doc-uk",',
            1,
        ),
        CFG_RAW.replace(
            '    "model_training_executed": false,',
            '    "model_training_executed": true,\n'
            '    "model_training_executed": false,',
            1,
        ),
        CFG_RAW.replace(
            '  "worker_id": "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN",',
            '  "\\u0077orker_id": "CONFLICT",\n'
            '  "worker_id": "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN",',
            1,
        ),
    ],
)
def test_duplicate_json_members_rejected_before_validation(raw):
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        load_strict_json_object(raw)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_nonfinite_json_numbers_rejected_before_validation(token):
    raw = CFG_RAW.replace(
        '"normalized_source_bytes": 48675',
        f'"normalized_source_bytes": {token}',
        1,
    )
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_strict_json_object(raw)


def test_non_object_json_root_rejected_before_validation():
    with pytest.raises(ValueError, match="root must be an object"):
        load_strict_json_object("[]")
