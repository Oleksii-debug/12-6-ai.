from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_next100_105_cross_source_dedup_v4.py"
SPEC = importlib.util.spec_from_file_location("next100_105_dedup_v4", SCRIPT)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _authorities() -> tuple[dict, dict, dict]:
    return (
        _load(ROOT / "configs/data/next100_105_cross_source_dedup_v4.json"),
        _load(ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"),
        _load(ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"),
    )


def test_expanded_inventory_binds_exact_converged_numeric_vector() -> None:
    config, base, convergence = _authorities()
    inventory = mod.build_expanded_inventory(config, base, convergence)
    assert len(inventory["sources"]) == 21
    assert sum(row["declared_capacity_bytes"] for row in inventory["sources"]) == 314_140

    by_modality = {
        modality: sum(
            row["declared_capacity_bytes"]
            for row in inventory["sources"]
            if row["modality"] == modality
        )
        for modality in ("uk", "en", "code")
    }
    assert by_modality == {"uk": 100_856, "en": 144_151, "code": 69_133}

    families = {
        modality: {
            row["source_family"]
            for row in inventory["sources"]
            if row["modality"] == modality
        }
        for modality in ("uk", "en", "code")
    }
    assert {key: len(value) for key, value in families.items()} == {"uk": 4, "en": 2, "code": 4}


def test_late_capacity_is_exactly_authority_delta() -> None:
    config, _, _ = _authorities()
    late = config["late_sources"]
    assert len(late) == 10
    assert sum(row["declared_capacity_bytes"] for row in late) == 70_170
    assert sum(row["declared_capacity_bytes"] for row in late if row["modality"] == "uk") == 10_812
    assert sum(row["declared_capacity_bytes"] for row in late if row["modality"] == "en") == 59_358
    assert not [row for row in late if row["modality"] == "code"]


def test_cp_python_zero_credit_source_is_not_silently_composed() -> None:
    config, _, convergence = _authorities()
    late_ids = {row["source_id"] for row in config["late_sources"]}
    assert not any("python" in source_id.lower() for source_id in late_ids)
    python_authorities = [
        item for item in convergence["late_authorities"]
        if item["worker_id"] == "NEXT100-037-DATA-EN-PYTHON-DOCS"
    ]
    assert len(python_authorities) == 1
    assert python_authorities[0]["numeric_capacity_bytes"] == 0
    assert python_authorities[0]["independent_family_credit"] == 0


def test_contract_fails_if_training_or_paid_compute_is_claimed() -> None:
    config, base, convergence = _authorities()
    for key in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        mutated = copy.deepcopy(config)
        mutated[key] = True
        with pytest.raises(mod.DedupV4Error):
            mod.build_expanded_inventory(mutated, base, convergence)


def test_contract_fails_on_capacity_or_source_count_drift() -> None:
    config, base, convergence = _authorities()
    mutated = copy.deepcopy(config)
    mutated["late_sources"][0]["declared_capacity_bytes"] += 1
    with pytest.raises(mod.DedupV4Error):
        mod.build_expanded_inventory(mutated, base, convergence)

    mutated = copy.deepcopy(config)
    mutated["late_sources"].pop()
    with pytest.raises(mod.DedupV4Error):
        mod.build_expanded_inventory(mutated, base, convergence)


def test_pinned_text_materializer_canonicalizes_one_terminal_lf_and_hashes() -> None:
    payload = b"bounded source\n"
    row = {
        "source_id": "fixture",
        "acquisition_url": "https://example.invalid/fixture",
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": hashlib.sha256(payload).hexdigest(),
    }
    with mock.patch.object(mod, "_download", return_value=b"bounded source"):
        assert mod._canonical_pinned_text(row) == payload


def test_pinned_text_materializer_fails_closed_on_hash_drift() -> None:
    row = {
        "source_id": "fixture",
        "acquisition_url": "https://example.invalid/fixture",
        "expected_raw_bytes": len(b"bounded source\n"),
        "expected_raw_sha256": "0" * 64,
    }
    with mock.patch.object(mod, "_download", return_value=b"bounded source"):
        with pytest.raises(mod.DedupV4Error):
            mod._canonical_pinned_text(row)


def test_nist_normalization_is_deterministic_and_redacts_email() -> None:
    source = " Header\r\n\r\n\r\nContact Alice@example.org   \f Body \n"
    first = mod._normalize_nist_extracted(source)
    second = mod._normalize_nist_extracted(source)
    assert first == second
    assert b"Alice@example.org" not in first
    assert b"<EMAIL>" in first
    assert first.endswith(b"\n")


def test_unknown_materializer_fails_closed() -> None:
    with pytest.raises(mod.DedupV4Error):
        mod.materialize_late_source({"source_id": "fixture", "materializer": "UNKNOWN"})
