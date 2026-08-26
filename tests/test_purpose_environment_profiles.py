from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "verify_purpose_environment.py"
SPEC = importlib.util.spec_from_file_location("verify_purpose_environment", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load purpose environment verifier")
ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENV)

EXPECTED_PROFILES = {
    "linux-x86_64-tokenizer-experiment",
    "linux-x86_64-transformers-interop",
    "linux-x86_64-cuda-training",
    "windows-x86_64-runtime",
}
CANONICAL_INDEX_FILE_SHA256 = "61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac"
CANONICAL_INDEX_SEMANTIC_SHA256 = "5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341"


def _canonical_hash(value: dict[str, object]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def test_registry_binds_frozen_canonical_lock_and_exact_profile_set() -> None:
    state = ENV.validate_registry(ROOT, "linux-x86_64-tokenizer-experiment")
    assert set(state["index"]["profiles"]) == EXPECTED_PROFILES
    assert state["canonical_ref"] == {
        "path": "requirements/locks/index.json",
        "file_sha256": CANONICAL_INDEX_FILE_SHA256,
        "index_sha256": CANONICAL_INDEX_SEMANTIC_SHA256,
    }
    assert state["profile"]["direct_requirements"] == ["tokenizers==0.23.1"]


def test_profiles_remain_purpose_specific() -> None:
    tokenizer = ENV.validate_registry(ROOT, "linux-x86_64-tokenizer-experiment")["profile"]
    transformers = ENV.validate_registry(ROOT, "linux-x86_64-transformers-interop")["profile"]
    cuda = ENV.validate_registry(ROOT, "linux-x86_64-cuda-training")["profile"]
    windows = ENV.validate_registry(ROOT, "windows-x86_64-runtime")["profile"]

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tokenizers==0.23.1" not in pyproject
    assert "transformers==5.15.1" not in pyproject
    assert tokenizer["kind"] == "linux-overlay"
    assert transformers["kind"] == "linux-overlay"
    assert cuda["kind"] == "linux-base-role"
    assert cuda["locks"] == {}
    assert cuda["runtime_expectations"]["gpu_execution_required_for_runtime_pass"] is True
    assert windows["kind"] == "windows-runtime"
    assert windows["python"]["version"] == "3.11.9"


def test_profile_tamper_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    shutil.copytree(ROOT / "requirements", copied / "requirements")
    profile_path = (
        copied
        / "requirements"
        / "profiles"
        / "linux-x86_64-tokenizer-experiment"
        / "profile.json"
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["purpose"] = "tampered"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ENV.PurposeEnvironmentError, match="file hash mismatch"):
        ENV.validate_registry(copied, "linux-x86_64-tokenizer-experiment")


def test_index_cannot_drop_a_required_purpose_profile_even_with_new_self_hash(tmp_path: Path) -> None:
    copied = tmp_path / "repo"
    shutil.copytree(ROOT / "requirements", copied / "requirements")
    index_path = copied / "requirements" / "profiles" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["profiles"].pop("linux-x86_64-transformers-interop")
    payload = dict(index)
    payload.pop("index_sha256", None)
    index["index_sha256"] = _canonical_hash(payload)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ENV.PurposeEnvironmentError, match="profile set mismatch"):
        ENV.validate_registry(copied, "linux-x86_64-tokenizer-experiment")


def test_invalid_source_sha_is_rejected_before_install() -> None:
    with pytest.raises(ENV.PurposeEnvironmentError, match="source SHA"):
        ENV.verify_install(
            root=ROOT,
            profile_id="linux-x86_64-tokenizer-experiment",
            source_sha="deadbeef",
            evidence_out=None,
            project_wheel=None,
        )
