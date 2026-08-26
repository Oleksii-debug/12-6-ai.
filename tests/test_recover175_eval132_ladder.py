from __future__ import annotations

import math
from pathlib import Path

import pytest

from twelve_six import evaluation_ua_v1 as ua
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six import recover175_eval132_ladder as r175

ROOT = Path(__file__).resolve().parents[1]


def test_recover175_reuses_frozen_eval132_registration() -> None:
    registration = r175._verify_eval_registration(ROOT)
    assert registration["status"] == "PASS"
    assert registration["dataset_sha256"] == ua.DATASET_SHA256
    assert registration["d06_registry_sha256"] == ua.D06_REGISTRY_SHA256
    assert registration["reserved_variant_count"] == 432
    assert registration["held_out"] is True
    assert registration["future_training_allowed"] is False
    assert registration["allowed_uses"] == ["evaluation"]


def test_recover175_common_ladder_contract_is_exact_m150_family() -> None:
    assert m150.SCALE_ORDER == ("100k", "500k", "1m")
    assert [m150.model_spec(scale).parameter_count() for scale in m150.SCALE_ORDER] == [95_568, 467_808, 1_037_696]
    assert m150.init_spec().identity_sha256() == "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def test_tokenizer_normalized_diagnostic_uses_source_bytes_not_completion_count() -> None:
    rows = []
    for index in range(216):
        preferred_bytes = 2 + index % 3
        contrast_bytes = 3 + index % 2
        rows.append({
            "preferred": {
                "logprob_nats": -0.5 * preferred_bytes,
                "source_bytes": preferred_bytes,
                "byte_tokens": preferred_bytes,
            },
            "contrast": {
                "logprob_nats": -0.75 * contrast_bytes,
                "source_bytes": contrast_bytes,
                "byte_tokens": contrast_bytes,
            },
        })
    evaluation = {
        "items": rows,
        "overall": {
            "correct": 216,
            "n": 216,
            "accuracy": 1.0,
            "accuracy_wilson95": [0.98, 1.0],
            "mean_margin_nats_per_source_byte": 0.25,
            "median_margin_nats_per_source_byte": 0.25,
        },
    }
    normalized = r175._normalized_likelihood_diagnostics(evaluation)
    assert normalized["preferred"]["tokens_per_source_byte"] == 1.0
    assert normalized["contrast"]["tokens_per_source_byte"] == 1.0
    assert normalized["preferred"]["conditional_bpb"] == pytest.approx(0.5 / math.log(2.0))
    assert normalized["contrast"]["conditional_bpb"] == pytest.approx(0.75 / math.log(2.0))
    assert normalized["tokenizer_length_artifact"]["length_artifact_confounded"] is False


def test_recover175_refuses_source_sha_drift_before_evidence_use(tmp_path: Path) -> None:
    with pytest.raises(r175.Recover175Error, match="bound to the verified M150 source SHA"):
        r175.execute(ROOT, tmp_path / "missing", tmp_path / "out", "0" * 40)


def test_recover175_does_not_authorize_broad_proficiency() -> None:
    manifest = r175._read_json(ROOT / "data/evaluation/ua_raw_base_v1/manifest.json")
    assert manifest["interpretation"]["proficiency_claim_authorized"] is False
    assert manifest["task"]["instruction_following"] is False
