from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_runtime349_20m_transformers_contract.py"
EVIDENCE = ROOT / "evidence/runtime349/20m_transformers_contract_v1.json"


def _load_validator():
    spec = importlib.util.spec_from_file_location("runtime349_validator", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime349_evidence_is_fail_closed_and_self_consistent() -> None:
    with EVIDENCE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    validator = _load_validator()
    validator.validate(payload)

    assert payload["status"] == "BLOCKED_NO_PUBLISHED_PRIMARY_20M_MODELSPEC"
    assert payload["primary_20m"]["modelspec_status"] == "MISSING"
    assert payload["required_primary_20m_parity"]["status"] == "NOT_RUN"
    assert payload["verdict"]["exactly_representable"] is None
    assert payload["verdict"]["complete_logits_parity"] is None
    assert payload["maintained_standard_llama_path"]["second_exporter_added"] is False
