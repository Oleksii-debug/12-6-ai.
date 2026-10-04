from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/next100_025_data_gov_registry_snapshot.py"
CONFIG = ROOT / "configs/data/next100_025_derzhgeocadastre_open_registry_v1.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("next100_025_snapshot", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_english_address_field_is_excluded_from_candidate_text() -> None:
    module = _load_tool()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "address" in config["privacy"]["exclude_key_fragments"]

    text, rejected = module.safe_record_text(
        {
            "title": "Реєстр набору державних даних України",
            "address": "Kyiv private administrative address",
        },
        config,
    )

    assert "Реєстр набору державних даних України" in text
    assert "Kyiv private administrative address" not in text
    assert rejected["excluded_key"] == 1
