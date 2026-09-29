from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixture = _load_module(
    "g05_g06_preflight_fixture",
    _ROOT / "tests" / "test_materialize_d03_current_g05_g06_preflight_v1.py",
)
post = _load_module(
    "post_g05_g06_materialization_v1",
    _ROOT / "src" / "twelve_six" / "data" / "post_g05_g06_materialization_v1.py",
)


def _cjson(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def test_native_preflight_scopes_external_llm_nonclaim_and_remains_consumable():
    g05 = fixture._g05()
    g06 = fixture._g06()
    qualification = fixture._terminal_qualification(g06)
    result = fixture._build(
        g05=g05,
        g06=g06,
        terminal_qualification=qualification,
        expected_terminal_qualification_identity=qualification[
            "qualification_identity_sha256"
        ],
    )

    truth = result["truth_boundary"]
    assert "external_llm_or_api_used_for_data_or_intelligence" not in truth
    assert truth["current_corpus_external_llm_free_claimed_by_this_preflight"] is False
    assert truth["authorized_optimized_target_exposure"] == 0
    assert truth["tokenizer_fit_authorized"] is False
    assert truth["training_executed"] is False
    assert truth["learned_weights_created"] is False
    assert truth["final_test_outcomes_read"] is False
    assert truth["paid_compute_used"] is False

    core = dict(result)
    identity = core.pop("composition_preflight_identity_sha256")
    assert identity == hashlib.sha256(_cjson(core)).hexdigest()

    assert (
        post._validate_preflight(
            result,
            expected_identity=identity,
            expected_g05_identity=g05["execution_identity_sha256"],
            expected_g06_envelope_identity=g06["evidence_identity_sha256"],
            expected_g06_identity=g06["privacy_execution_authority"][
                "execution_identity_sha256"
            ],
            expected_g06_terminal_qualification_identity=qualification[
                "qualification_identity_sha256"
            ],
        )
        == fixture.INPUT_ROOT
    )
