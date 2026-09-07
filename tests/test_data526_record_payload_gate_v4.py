from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path("tools/validate_data526_record_payload_gate_v4.py")
CONFIG_PATH = Path("configs/data/research_corpus_v1_predecontam_blocker_v4.json")
spec = importlib.util.spec_from_file_location("data526_record_payload_gate_v4", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def rehash(doc: dict[str, object]) -> None:
    doc["evidence_identity_sha256"] = module.identity(doc)


def test_canonical_gate_passes() -> None:
    module.validate(load())


def test_source_object_identity_cannot_masquerade_as_record_payload_freeze() -> None:
    doc = load()
    gate = doc["record_materialization_gate"]
    gate["frozen"] = True
    gate["record_count"] = 35
    gate["record_inventory_digest_sha256"] = "5e54c3587d6d2c0a7d540a1e8aefae77522c97a41c0033a05e78ce22a8e7c617"
    rehash(doc)
    try:
        module.validate(doc)
    except ValueError as exc:
        assert "record freeze fabricated" in str(exc)
    else:
        raise AssertionError("source-object inventory was accepted as record payload freeze")


def test_decontamination_cannot_run_before_payload_bound_inventory() -> None:
    doc = load()
    doc["downstream_gates"]["reserved_evaluation_decontamination"] = "PERMITTED_ONLY_AGAINST_EXACT_CANDIDATE_IDENTITY"
    rehash(doc)
    try:
        module.validate(doc)
    except ValueError as exc:
        assert "decontamination permitted" in str(exc)
    else:
        raise AssertionError("decontamination gate failed open")


def test_source_bytes_cannot_become_optimized_targets() -> None:
    doc = load()
    doc["claim_boundary"]["authorized_unique_optimized_targets"] = 2215615
    rehash(doc)
    try:
        module.validate(doc)
    except ValueError as exc:
        assert "downstream scientific gate fabricated" in str(exc)
    else:
        raise AssertionError("source bytes were accepted as optimized-target capacity")


def test_terminal_dedup_binding_is_exact() -> None:
    doc = copy.deepcopy(load())
    doc["global_dedup_terminal"]["head_sha"] = "0" * 40
    rehash(doc)
    try:
        module.validate(doc)
    except ValueError as exc:
        assert "terminal dedup authority drift" in str(exc)
    else:
        raise AssertionError("dedup authority drift accepted")
