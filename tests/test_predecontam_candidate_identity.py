from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(module_name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load tool module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_tool("build_predecontam_candidate_identity", "build_predecontam_candidate_identity.py")
validator = _load_tool(
    "validate_predecontam_candidate_identity", "validate_predecontam_candidate_identity.py"
)


def _candidate() -> dict:
    records = []
    index = 0
    for stratum in builder.STRATA:
        for suffix in ("a", "b"):
            index += 1
            records.append(
                builder._synthetic_record(index, stratum, f"{stratum}.family.{suffix}")
            )
    return builder.build_candidate({"schema": builder.INPUT_SCHEMA, "records": records})


def test_validator_accepts_exact_builder_output() -> None:
    report = validator.validate_candidate(_candidate())
    assert report["verdict"] == "PASS"
    assert report["record_count"] == 6
    assert report["final_training_authorized"] is False
    assert report["decontamination_executed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_count", 999),
        ("total_normalized_utf8_bytes", 1),
        ("source_authority_bundle_identity_sha256", "0" * 64),
        ("candidate_record_inventory_identity_sha256", "0" * 64),
        ("candidate_identity_sha256", "0" * 64),
        ("final_training_authorized", True),
        ("decontamination_executed", True),
        ("replay_authorized", True),
    ],
)
def test_validator_rejects_tampered_top_level_fields(field: str, value: object) -> None:
    candidate = _candidate()
    candidate[field] = value
    with pytest.raises(ValueError):
        validator.validate_candidate(candidate)


def test_validator_rejects_noncanonical_record_order() -> None:
    candidate = _candidate()
    candidate["records"] = list(reversed(candidate["records"]))
    with pytest.raises(ValueError, match="canonical builder order"):
        validator.validate_candidate(candidate)


def test_validator_rejects_duplicate_source_identity_even_if_candidate_hash_is_stale() -> None:
    candidate = _candidate()
    tampered = copy.deepcopy(candidate["records"][0])
    tampered["normalized_sha256"] = "f" * 64
    candidate["records"].append(tampered)
    with pytest.raises(ValueError):
        validator.validate_candidate(candidate)


def test_validator_rejects_record_contract_extension() -> None:
    candidate = _candidate()
    candidate["records"][0]["unexpected_authority"] = "should-not-be-silently-accepted"
    with pytest.raises(ValueError, match="field mismatch"):
        validator.validate_candidate(candidate)
