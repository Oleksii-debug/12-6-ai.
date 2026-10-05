from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "run_d03_nbu_current40_global_dedup_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_d03_nbu_current40_global_dedup_test", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carrier_binds_current_main_and_current40_authority() -> None:
    mod = _load()
    assert mod.EXPECTED_MAIN == "019944d5fe12334791f05f1232d13de4a12e37d3"
    assert mod.EXECUTION_CLAIM == 548
    assert mod.EXECUTION_PR == 2809
    assert mod.EXPECTED_NBU_OBJECTS == 40
    assert mod.EXPECTED_NBU_BYTES == 891_494
    assert mod.EXPECTED_COMBINED_OBJECTS == 303
    assert mod.EXPECTED_COMBINED_BYTES == 6_985_459
    assert mod.INTAKE_PATH == "src/twelve_six/data/nbu_current40_dedup_intake.py"
    assert mod.CARRIER_PATH == "tools/run_d03_nbu_current40_global_dedup_v1.py"
    assert mod.PHYSICAL_AUTHORITY_PATH in mod.PRODUCT_PATHS
    assert mod.INTAKE_PATH in mod.PRODUCT_PATHS
    assert mod.CARRIER_PATH in mod.PRODUCT_PATHS
    assert mod.INTAKE_PATH not in mod.MAIN_AUTHORITY_PATHS


def test_carrier_uses_incumbent_indexed_executor_not_new_matcher() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "indexed.attest_incumbent_runtime(matcher)" in source
    assert "indexed.audit_payloads_indexed(" in source
    assert "matcher.verify_report(indexed_report)" in source
    assert "from twelve_six.data import incumbent_dedup_indexed_execution as indexed" in source


def test_selected_execution_head_must_equal_observed_head(monkeypatch) -> None:
    mod = _load()
    selected = "a" * 40
    monkeypatch.setattr(
        mod,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=selected + "\n",
            stderr="",
        ),
    )
    assert mod._bind_execution_head(selected) == selected


@pytest.mark.parametrize("bad", ["", "a" * 39, "A" * 40, "g" * 40])
def test_selected_execution_head_rejects_malformed_sha(bad: str) -> None:
    mod = _load()
    with pytest.raises(mod.NbuGlobalDedupError, match="exact lowercase 40-hex"):
        mod._bind_execution_head(bad)


@pytest.mark.parametrize("bad", [0, -1, False, 1.5, "10"])
def test_work_budgets_require_exact_positive_ints(bad: object) -> None:
    mod = _load()
    with pytest.raises(mod.NbuGlobalDedupError, match="must be exact positive int"):
        mod._validate_work_budgets(bad, 1, 1)


def test_source_admission_receipt_remains_zero_credit() -> None:
    mod = _load()
    receipt = {
        "truth_boundary": {
            "whole_corpus_external_llm_cleanliness_claimed": False,
        }
    }
    scope = mod._source_admission_provenance_scope(receipt)
    assert scope["current_corpus_external_llm_free_claimed_by_this_carrier"] is False
    assert scope["source_local_negative_promoted_as_corpus_global_truth"] is False
