from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import twelve_six.data.post_g05_g06_materialization_v2 as materializer_v2_module
from twelve_six.data.external_llm_provenance_quarantine_v1 import (
    EXPECTED_AUTHORITY_IDENTITY_SHA256,
)
from twelve_six.data.post_g05_g06_materialization_v2 import (
    PostG05G06MaterializationV2Error,
    materialize_post_g05_g06_v2,
)

ROOT = Path(__file__).resolve().parents[1]
HELPERS_PATH = ROOT / "tests/test_post_g05_g06_materialization_v1.py"
SPEC = importlib.util.spec_from_file_location("post_g05_g06_v1_test_helpers", HELPERS_PATH)
assert SPEC is not None and SPEC.loader is not None
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
AUTHORITY = json.loads(
    (ROOT / "configs/data/d03_external_llm_provenance_quarantine_v1.json").read_text(
        encoding="utf-8"
    )
)


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw,
        usedforsecurity=False,
    ).hexdigest()


def running_v2_blob() -> str:
    return blob_sha(Path(materializer_v2_module.__file__).read_bytes())


def v2_case(tmp_path: Path):
    kwargs = helpers.setup_case(tmp_path)
    kwargs.update(
        {
            "provenance_quarantine_authority": AUTHORITY,
            "expected_provenance_quarantine_identity_sha256": (
                EXPECTED_AUTHORITY_IDENTITY_SHA256
            ),
            "expected_materializer_v2_implementation_git_blob_sha1": running_v2_blob(),
        }
    )
    return kwargs


def test_v2_materializes_clean_synthetic_case_without_false_external_llm_claim(
    tmp_path,
) -> None:
    kwargs = v2_case(tmp_path)
    out_a, inventory_a, evidence_a = materialize_post_g05_g06_v2(**kwargs)
    out_b, inventory_b, evidence_b = materialize_post_g05_g06_v2(**kwargs)
    assert out_a == out_b
    assert inventory_a == inventory_b
    assert evidence_a == evidence_b
    assert evidence_a["schema_version"] == "12-6.d03-post-g05-g06-materialization.v2"
    assert "external_llm_or_api_used_for_data_or_intelligence" not in evidence_a[
        "truth_boundary"
    ]
    assert evidence_a["provenance_guard"] == {
        "known_external_llm_contamination_absent": True,
        "quarantine_identity_sha256": EXPECTED_AUTHORITY_IDENTITY_SHA256,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "successor_authority_rebuild_required_for_invalidated_v5_v6_v8": True,
    }
    assert (
        evidence_a["input"]["external_llm_provenance_quarantine_identity_sha256"]
        == EXPECTED_AUTHORITY_IDENTITY_SHA256
    )
    assert "SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED" in evidence_a[
        "remaining_materialization_blockers"
    ]
    assert evidence_a["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert evidence_a["truth_boundary"]["training_executed"] is False


def test_v2_rejects_known_nomis_identity_before_g05_g06_transform(tmp_path) -> None:
    kwargs = v2_case(tmp_path)
    kwargs["records"][0] = {
        **kwargs["records"][0],
        "record_id": "ua.verba.nomis1864.bounded24",
    }
    with pytest.raises(
        PostG05G06MaterializationV2Error,
        match="known Nomis1864 authority identity quarantined",
    ):
        materialize_post_g05_g06_v2(**kwargs)


def test_v2_rejects_self_resealed_or_substituted_quarantine_authority(tmp_path) -> None:
    kwargs = v2_case(tmp_path)
    authority = json.loads(json.dumps(AUTHORITY))
    authority["enforcement"]["all_other_corpus_bytes_declared_external_llm_clean"] = True
    kwargs["provenance_quarantine_authority"] = authority
    with pytest.raises(
        PostG05G06MaterializationV2Error,
        match="quarantine self-hash mismatch|quarantine enforcement drift",
    ):
        materialize_post_g05_g06_v2(**kwargs)


def test_v2_implementation_substitution_fails_before_transform(tmp_path) -> None:
    kwargs = v2_case(tmp_path)
    kwargs["expected_materializer_v2_implementation_git_blob_sha1"] = "f" * 40
    with pytest.raises(
        PostG05G06MaterializationV2Error,
        match="V2 materializer implementation Git blob drift",
    ):
        materialize_post_g05_g06_v2(**kwargs)
