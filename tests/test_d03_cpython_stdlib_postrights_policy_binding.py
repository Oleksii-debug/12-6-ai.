from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.d03_cpython_stdlib_postrights_admission import (
    EXPECTED_POLICY_IDENTITY_SHA256,
    ROOT_LICENSE_BLOB_SHA1,
    CPythonRightsAdmissionError,
    build_admission,
    git_blob_sha1,
    load_and_validate_policy,
)


def _row(path: str, text: str) -> dict[str, object]:
    raw = text.encode("utf-8")
    return {
        "record_id": f"cpython:{path}",
        "source_path": path,
        "family": "code.python.cpython",
        "modality": "code",
        "git_blob_sha1": git_blob_sha1(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "payload_bytes": len(raw),
        "text": text,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _tree(row: dict[str, object]) -> dict[str, str]:
    return {
        "LICENSE": ROOT_LICENSE_BLOB_SHA1,
        str(row["source_path"]): str(row["git_blob_sha1"]),
    }


def _policy() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return load_and_validate_policy(
        root / "configs/data/d03_cpython_stdlib_source_rights_v1.json"
    )


def test_canonical_consumed_policy_identity_is_emitted_from_verified_snapshot() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    policy = _policy()
    assert policy["policy_identity_sha256"] == EXPECTED_POLICY_IDENTITY_SHA256

    report = build_admission(
        [row],
        _tree(row),
        policy,
        require_historical_identity=False,
    )

    assert report["rights_authority"]["policy_identity_sha256"] == EXPECTED_POLICY_IDENTITY_SHA256


def test_build_admission_rejects_post_validation_marker_mutation() -> None:
    row = _row(
        "Lib/example.py",
        "# SPDX-License-Identifier: MIT\nvalue = 3\n",
    )
    forged = deepcopy(_policy())
    forged["policy"]["direct_notice_markers"] = []

    with pytest.raises(CPythonRightsAdmissionError, match="rights policy identity drift"):
        build_admission(
            [row],
            _tree(row),
            forged,
            require_historical_identity=False,
        )


def test_build_admission_rejects_post_validation_root_default_mutation() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    forged = deepcopy(_policy())
    forged["policy"]["root_default_license"] = "MIT"

    with pytest.raises(CPythonRightsAdmissionError, match="rights policy identity drift"):
        build_admission(
            [row],
            _tree(row),
            forged,
            require_historical_identity=False,
        )
