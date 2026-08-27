from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.liger_kernel_qualification import (
    SUPPORTED_OPERATORS,
    canonical_sha256,
    compare_numeric_sequences,
    deterministic_probe_input,
    repeatability_digest,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "configs/research/liger_kernel_qualification_v1.json").read_text(encoding="utf-8")
)


def test_manifest_is_valid_and_deterministic() -> None:
    validate_manifest(MANIFEST)
    assert canonical_sha256(MANIFEST) == canonical_sha256(copy.deepcopy(MANIFEST))


def test_exact_probe_repeatability() -> None:
    probe = deterministic_probe_input("RMSNorm", [2, 8], "float32")
    assert repeatability_digest(probe) == repeatability_digest(copy.deepcopy(probe))


@pytest.mark.parametrize("operator", SUPPORTED_OPERATORS)
def test_all_required_operator_names_are_accepted(operator: str) -> None:
    assert deterministic_probe_input(operator, [1, 4])["operator"] == operator


def test_numeric_parity_accepts_within_tolerance() -> None:
    assert compare_numeric_sequences([1.0, 2.0], [1.000001, 1.999999], 1e-5, 1e-6)


def test_numeric_parity_rejects_length_drift() -> None:
    assert not compare_numeric_sequences([1.0], [1.0, 2.0], 1e-5, 1e-6)


def test_numeric_parity_rejects_non_finite_output() -> None:
    assert not compare_numeric_sequences([1.0], [float("nan")], 1e-5, 1e-6)


def test_invalid_tolerance_fails_closed() -> None:
    with pytest.raises(ValueError):
        compare_numeric_sequences([1.0], [1.0], float("nan"), 1e-6)


def test_shape_rejects_bool_and_nonpositive() -> None:
    with pytest.raises(ValueError):
        deterministic_probe_input("RoPE", [True, 4])
    with pytest.raises(ValueError):
        deterministic_probe_input("RoPE", [0, 4])


def test_manifest_rejects_upstream_drift() -> None:
    bad = copy.deepcopy(MANIFEST)
    bad["upstream"]["commit"] = "0" * 40
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_manifest_rejects_fake_promotion() -> None:
    bad = copy.deepcopy(MANIFEST)
    bad["promotion_state"] = "PARITY_PROVEN"
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_manifest_rejects_canonical_base_dependency() -> None:
    bad = copy.deepcopy(MANIFEST)
    bad["canonical_base_dependency"] = True
    with pytest.raises(ValueError):
        validate_manifest(bad)
