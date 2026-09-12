from __future__ import annotations

from types import SimpleNamespace

import pytest

from twelve_six.data import incumbent_dedup_indexed_execution as indexed


EXPECTED_DATA232_GIT_BLOB_SHA1 = "dab5da98dfc43133aa8f3c2e3c78c809252b741b"
EXPECTED_THRESHOLDS = {
    "natural_shingle_tokens": 3,
    "natural_near_jaccard": 0.80,
    "natural_fragment_containment": 0.88,
    "natural_fragment_min_tokens": 18,
    "code_shingle_tokens": 7,
    "code_near_jaccard": 0.86,
    "code_fragment_containment": 0.90,
    "code_fragment_min_tokens": 16,
    "code_copy_jaccard": 0.82,
    "code_copy_min_tokens": 16,
}


def _runtime(tmp_path, *, thresholds: dict[str, int | float]):
    v1_path = tmp_path / "incumbent_v1.py"
    v3_path = tmp_path / "incumbent_v3.py"
    v1_path.write_text("# exact-file identity stand-in for independent attestation test\n")
    v3_path.write_text("# exact-file identity stand-in for independent attestation test\n")

    def _callable(*args, **kwargs):
        del args, kwargs
        return None

    v1 = SimpleNamespace(
        __file__=str(v1_path),
        DEFAULT_THRESHOLDS=dict(thresholds),
        _validate_inventory=_callable,
        _pair_matches=_callable,
        _sha256=_callable,
        _canonical_bytes=_callable,
    )
    v3 = SimpleNamespace(
        __file__=str(v3_path),
        v1=v1,
        _validate_inventory=_callable,
        _fingerprint=_callable,
        _lineage_matches=_callable,
        _summary_for_ids=_callable,
    )
    expected_v1_blob = indexed._git_blob_sha1(v1_path.read_bytes())
    expected_v3_blob = indexed._git_blob_sha1(v3_path.read_bytes())
    return v3, expected_v3_blob, expected_v1_blob


def test_attestation_binds_terminal_data232_dependency_blob():
    """Execution authority must bind V3 + V1 + the transitive DATA-232 matcher closure."""
    assert (
        getattr(indexed, "EXPECTED_DATA232_GIT_BLOB_SHA1", None)
        == EXPECTED_DATA232_GIT_BLOB_SHA1
    )


def test_attestation_rejects_positive_but_noncanonical_threshold_vector(tmp_path):
    """Positive/in-range is insufficient: exact #824 thresholds are binding science."""
    drifted = dict(EXPECTED_THRESHOLDS)
    drifted["natural_near_jaccard"] = 0.81
    v3, expected_v3_blob, expected_v1_blob = _runtime(tmp_path, thresholds=drifted)

    with pytest.raises(indexed.IndexedExecutionError, match="threshold"):
        indexed.attest_incumbent_runtime(
            v3,
            expected_v3_blob=expected_v3_blob,
            expected_v1_blob=expected_v1_blob,
        )


def test_attestation_accepts_exact_terminal_threshold_vector(tmp_path):
    """The independent regression also proves the exact vector remains admissible."""
    v3, expected_v3_blob, expected_v1_blob = _runtime(
        tmp_path,
        thresholds=EXPECTED_THRESHOLDS,
    )
    indexed.attest_incumbent_runtime(
        v3,
        expected_v3_blob=expected_v3_blob,
        expected_v1_blob=expected_v1_blob,
    )
