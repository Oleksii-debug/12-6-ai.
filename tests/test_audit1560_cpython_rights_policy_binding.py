from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.d03_cpython_stdlib_postrights_admission import (
    CPythonRightsAdmissionError,
    EXPECTED_POLICY_IDENTITY_SHA256,
    ROOT_LICENSE_BLOB_SHA1,
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


def _validated_policy() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return load_and_validate_policy(
        root / "configs/data/d03_cpython_stdlib_source_rights_v1.json"
    )


def test_build_admission_rejects_post_validation_policy_substitution() -> None:
    row = _row(
        "Lib/example.py",
        "# SPDX-License-Identifier: MIT\nvalue = 3\n",
    )
    policy = _validated_policy()
    forged = deepcopy(policy)
    forged["policy"]["direct_notice_markers"] = []

    with pytest.raises(CPythonRightsAdmissionError, match="rights policy identity drift"):
        build_admission(
            [row],
            _tree(row),
            forged,
            require_historical_identity=False,
        )


def test_build_admission_cannot_relabel_mutated_policy_as_canonical() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    policy = _validated_policy()
    forged = deepcopy(policy)
    forged["policy"]["root_default_license"] = "MIT"

    try:
        report = build_admission(
            [row],
            _tree(row),
            forged,
            require_historical_identity=False,
        )
    except CPythonRightsAdmissionError:
        return

    assert report["rights_authority"]["policy_identity_sha256"] != EXPECTED_POLICY_IDENTITY_SHA256, (
        "candidate-controlled mutated policy was accepted while evidence still claimed the "
        "canonical expected policy identity"
    )
