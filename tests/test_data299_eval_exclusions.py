from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "materialize_data299_eval_exclusions.py"
REGISTRY_PATH = ROOT / "data" / "evaluation" / "data299_eval_exclusion_registry_v1.json"

spec = importlib.util.spec_from_file_location("data299_materializer", TOOL_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_registry_identity_and_public_boundary() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    mod._verify_registry(registry)
    assert registry["execution_class"] == "LOCAL_FREE"
    assert registry["evaluation_access_boundary"] == "AUTHORITY_METADATA_AND_HASHES_ONLY"
    assert registry["construction_gate"]["public_evidence_hash_only"] is True


def test_selection_validation_bindings() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    ua = registry["selection_validation"]["ua"]
    en = registry["selection_validation"]["en"]
    code = registry["selection_validation"]["code"]
    assert ua["terminal_authority_bound"] is False
    assert ua["status"] == "NONTERMINAL_HEAD_NOT_ADMITTED"
    assert ua["exact_sha256"] == []
    assert en["terminal_authority_bound"] is True
    assert en["exact_count"] == 2
    assert len(en["exact_sha256"]) == 2
    assert code["terminal_authority_bound"] is True
    assert code["exact_count"] == 0
    assert code["exact_sha256"] == []


def test_expected_hash_shards_and_memorization() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    by_id = {item["authority_id"]: item for item in registry["reserved_authorities"]}
    assert by_id["EVAL132-UA-diagnostic"]["hash_shard"]["expected_count"] == 432
    assert by_id["EVAL133-EN-diagnostic"]["hash_shard"]["expected_count"] == 32
    assert by_id["EVAL134-code-diagnostic"]["hash_shard"]["expected_count"] == 62
    mem = by_id["EVAL136-memorization-reservation"]
    assert mem["exact_count"] == 54
    assert len(mem["exact_sha256"]) == 54
    assert mem["near_match_regeneration_required"] is True


def test_data232_near_match_binding_is_exact() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    near = registry["near_match_authority"]
    assert near["required"] is True
    assert near["method_id"] == "data232-deterministic-overlap-cluster-v2"
    assert near["source_commit"] == "4b9d7c2e1dc2806ca91e33e6570630b3f43af24a"
    assert near["matcher_git_blob_sha1"] == "dab5da98dfc43133aa8f3c2e3c78c809252b741b"
    assert near["config_git_blob_sha1"] == "993338ad73441ac4019f766bab21f763b1dd7947"
