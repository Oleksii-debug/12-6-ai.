from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_d03_ua_nbu_pdftotext_v1.py"
CONFIG = ROOT / "configs" / "data" / "d03_ua_nbu_pdftotext_materialization_v1.json"

spec = importlib.util.spec_from_file_location("nbu_pdftotext_authority", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _canonical_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _reseal(config: dict) -> dict:
    config["contract_identity_sha256"] = module.self_identity(
        config, "contract_identity_sha256"
    )
    return config


def test_canonical_config_is_unchanged_and_authoritative() -> None:
    config = module.load_config(CONFIG)
    assert config["contract_identity_sha256"] == (
        "c7e3222fc609cbe22d255a1732bd8130ce21d50185a9ef48bbb92f1d5ab288cf"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("base_authority", "parent_head_sha"), "0" * 40),
        (("base_authority", "discovery_head_sha"), "1" * 40),
        (
            ("base_authority", "parent_pdf_pin_contract_identity_sha256"),
            "2" * 64,
        ),
        (
            ("base_authority", "discovery_contract_identity_sha256"),
            "3" * 64,
        ),
        (("source", "document_path_regex"), r"^/ua/legislation/.*$"),
        (("source", "official_pdf_path_regex"), r"^/.*\.pdf$"),
        (("materialization", "hard_max_records"), 239),
        (("materialization", "min_text_bytes"), 31),
        (("materialization", "source_fetch_timeout_seconds"), 31),
    ],
)
def test_resealed_authority_and_policy_mutations_fail_closed(path, value) -> None:
    config = deepcopy(_canonical_config())
    config[path[0]][path[1]] = value
    _reseal(config)
    with pytest.raises(module.NbuTextMaterializationError, match="config .* drift"):
        module.validate_config(config)


def test_unknown_fields_and_bool_int_aliases_fail_closed() -> None:
    config = deepcopy(_canonical_config())
    config["unexpected_authority"] = "not allowed"
    _reseal(config)
    with pytest.raises(module.NbuTextMaterializationError, match="key-set drift"):
        module.validate_config(config)

    alias = deepcopy(_canonical_config())
    alias["claims"]["optimizer_updates"] = False
    _reseal(alias)
    with pytest.raises(module.NbuTextMaterializationError, match="type drift"):
        module.validate_config(alias)


def test_duplicate_root_and_nested_json_keys_fail_closed(tmp_path: Path) -> None:
    canonical = CONFIG.read_text(encoding="utf-8")
    root_duplicate = canonical.replace(
        '{\n  "schema":',
        '{\n  "schema": "duplicate",\n  "schema":',
        1,
    )
    nested_duplicate = canonical.replace(
        '    "repository": "Oleksii-debug/12-6-ai.",',
        '    "repository": "duplicate",\n'
        '    "repository": "Oleksii-debug/12-6-ai.",',
        1,
    )

    for index, payload in enumerate((root_duplicate, nested_duplicate)):
        path = tmp_path / f"duplicate-{index}.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(module.NbuTextMaterializationError, match="duplicate JSON key"):
            module.load_config(path)
