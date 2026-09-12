from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.d03_cpython_stdlib_postrights_admission import (
    EXPECTED_POLICY_IDENTITY_SHA256,
    HISTORICAL_INVENTORY_IDENTITY_SHA256,
    ROOT_LICENSE_BLOB_SHA1,
    UPSTREAM_TREE,
    CPythonRightsAdmissionError,
    build_admission,
    candidate_inventory_identity,
    classify_row,
    git_blob_sha1,
    load_and_validate_policy,
    parse_pinned_tree_response,
    validate_historical_candidate,
    validate_repository_bindings,
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


def _tree(*rows: dict[str, object], extra: dict[str, str] | None = None) -> dict[str, str]:
    result = {"LICENSE": ROOT_LICENSE_BLOB_SHA1}
    for row in rows:
        result[str(row["source_path"])] = str(row["git_blob_sha1"])
    if extra:
        result.update(extra)
    return result


def _policy() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    return load_and_validate_policy(
        root / "configs/data/d03_cpython_stdlib_source_rights_v1.json"
    )


def test_policy_identity_is_exact_and_zero_credit() -> None:
    policy = _policy()
    raw = {
        key: value for key, value in policy.items() if key != "policy_identity_sha256"
    }
    identity = hashlib.sha256(
        (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    assert identity == EXPECTED_POLICY_IDENTITY_SHA256
    assert policy["truth_boundary"]["training_authorized_bytes"] == 0
    assert policy["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_notice_free_file_without_local_override_is_admitted() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    decision = classify_row(row, _tree(row), _policy())
    assert decision["admitted"] is True
    assert decision["license_id"] == "PSF-2.0"
    assert decision["reason"] == "PINNED_ROOT_PSF2_DEFAULT_NO_LOCAL_OVERRIDE_OBSERVED"


def test_direct_rights_notice_fails_closed() -> None:
    row = _row("Lib/example.py", "# Copyright 2026 Example\nvalue = 3\n")
    decision = classify_row(row, _tree(row), _policy())
    assert decision["admitted"] is False
    assert decision["license_id"] is None
    assert decision["reason"] == "DIRECT_RIGHTS_NOTICE_PRESENT"


def test_direct_rights_notice_after_evidence_window_fails_closed() -> None:
    text = "# " + ("x" * 12_001) + "\n# SPDX-License-Identifier: MIT\nvalue = 3\n"
    row = _row("Lib/example.py", text)
    decision = classify_row(row, _tree(row), _policy())
    assert decision["admitted"] is False
    assert decision["license_id"] is None
    assert decision["reason"] == "DIRECT_RIGHTS_NOTICE_PRESENT"
    assert decision["direct_notice_marker"] == "spdx-license-identifier"


def test_ancestor_license_marker_fails_closed() -> None:
    row = _row("Lib/example/module.py", "value = 3\n")
    decision = classify_row(
        row,
        _tree(row, extra={"Lib/example/LICENSE": "1" * 40}),
        _policy(),
    )
    assert decision["admitted"] is False
    assert decision["reason"] == "ANCESTOR_RIGHTS_MARKER_PRESENT"
    assert decision["ancestor_rights_marker_path"] == "Lib/example/LICENSE"


def test_candidate_tamper_and_bool_alias_fail_closed() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    tampered = deepcopy(row)
    tampered["payload_bytes"] = True
    with pytest.raises(CPythonRightsAdmissionError, match="payload bytes invalid"):
        classify_row(tampered, _tree(row), _policy())

    tampered = deepcopy(row)
    tampered["text"] = "value = 4\n"
    with pytest.raises(CPythonRightsAdmissionError, match="payload sha drift"):
        classify_row(tampered, _tree(row), _policy())


def test_text_free_report_and_closed_world_decisions() -> None:
    admitted = _row("Lib/a.py", "a = 1\n")
    held = _row("Lib/b.py", "# license: custom\nb = 2\n")
    rows = [admitted, held]
    report = build_admission(
        rows,
        _tree(admitted, held),
        _policy(),
        require_historical_identity=False,
    )
    assert report["source_admission"]["admitted_objects"] == 1
    assert report["source_admission"]["held_objects"] == 1
    assert report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0
    assert report["truth_boundary"]["training_authorized_bytes"] == 0
    assert '"text"' not in json.dumps(report, sort_keys=True)
    assert report["candidate"]["inventory_identity_sha256"] == candidate_inventory_identity(rows)


def test_complete_tree_response_is_required() -> None:
    payload = {
        "sha": UPSTREAM_TREE,
        "truncated": False,
        "tree": [
            {"path": "LICENSE", "type": "blob", "sha": ROOT_LICENSE_BLOB_SHA1},
            {"path": "Lib/a.py", "type": "blob", "sha": "2" * 40},
        ],
    }
    assert parse_pinned_tree_response(payload)["LICENSE"] == ROOT_LICENSE_BLOB_SHA1
    payload["truncated"] = True
    with pytest.raises(CPythonRightsAdmissionError, match="must be complete"):
        parse_pinned_tree_response(payload)


def test_fake_candidate_cannot_claim_historical_inventory() -> None:
    row = _row("Lib/a.py", "a = 1\n")
    assert candidate_inventory_identity([row]) != HISTORICAL_INVENTORY_IDENTITY_SHA256
    with pytest.raises(CPythonRightsAdmissionError, match="selected object count drift"):
        validate_historical_candidate([row], _tree(row))


def test_exact_merged_repository_bindings_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "reports/d03/cpython_stdlib_terminal_execution_v1.json").exists():
        pytest.skip("standalone focused-test checkout does not contain merged lineage")
    validate_repository_bindings(root)
