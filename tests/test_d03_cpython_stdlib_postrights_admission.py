from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

import twelve_six.d03_cpython_stdlib_postrights_admission as rights
from twelve_six.d03_cpython_stdlib_postrights_admission import (
    EXPECTED_POLICY_IDENTITY_SHA256,
    HISTORICAL_INVENTORY_IDENTITY_SHA256,
    ROOT_LICENSE_BLOB_SHA1,
    UPSTREAM_TREE,
    CPythonRightsAdmissionError,
    PinnedTreeAuthority,
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
    assert decision["direct_notice_marker"] is not None


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


def _synthetic_complete_authority(
    monkeypatch: pytest.MonkeyPatch,
    *rows: dict[str, object],
    extra: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], PinnedTreeAuthority]:
    entries: list[dict[str, object]] = [
        {
            "path": "LICENSE",
            "mode": "100644",
            "type": "blob",
            "sha": ROOT_LICENSE_BLOB_SHA1,
            "size": 13_804,
        }
    ]
    for row in rows:
        entries.append(
            {
                "path": row["source_path"],
                "mode": "100644",
                "type": "blob",
                "sha": row["git_blob_sha1"],
                "size": row["payload_bytes"],
            }
        )
    if extra:
        entries.extend(extra)
    entries.sort(key=lambda item: str(item["path"]))
    semantics = {"entries": entries, "sha": UPSTREAM_TREE, "truncated": False}
    blob_projection = [
        {"path": item["path"], "sha": item["sha"]}
        for item in entries
        if item["type"] == "blob"
    ]
    semantics_bytes = rights.canonical_json_bytes(semantics)
    blob_bytes = rights.canonical_json_bytes(blob_projection)
    monkeypatch.setattr(rights, "EXPECTED_TREE_ENTRY_COUNT", len(entries))
    monkeypatch.setattr(
        rights,
        "EXPECTED_TREE_BLOB_COUNT",
        sum(item["type"] == "blob" for item in entries),
    )
    monkeypatch.setattr(
        rights,
        "EXPECTED_TREE_TREE_COUNT",
        sum(item["type"] == "tree" for item in entries),
    )
    monkeypatch.setattr(
        rights,
        "EXPECTED_TREE_COMMIT_COUNT",
        sum(item["type"] == "commit" for item in entries),
    )
    monkeypatch.setattr(rights, "EXPECTED_TREE_SEMANTICS_BYTES", len(semantics_bytes))
    monkeypatch.setattr(rights, "EXPECTED_TREE_BLOB_MAP_BYTES", len(blob_bytes))
    monkeypatch.setattr(
        rights,
        "EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256",
        rights.sha256_bytes(semantics_bytes),
    )
    monkeypatch.setattr(
        rights,
        "EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256",
        rights.sha256_bytes(blob_bytes),
    )
    payload: dict[str, object] = {
        "sha": UPSTREAM_TREE,
        "truncated": False,
        "tree": deepcopy(entries),
    }
    return payload, parse_pinned_tree_response(payload)


def test_incomplete_tree_response_is_rejected_even_with_root_license() -> None:
    payload = {
        "sha": UPSTREAM_TREE,
        "truncated": False,
        "tree": [
            {
                "path": "LICENSE",
                "mode": "100644",
                "type": "blob",
                "sha": ROOT_LICENSE_BLOB_SHA1,
                "size": 13_804,
            },
            {
                "path": "Lib/a.py",
                "mode": "100644",
                "type": "blob",
                "sha": "2" * 40,
                "size": 5,
            },
        ],
    }
    with pytest.raises(CPythonRightsAdmissionError, match="complete tree entry count drift"):
        parse_pinned_tree_response(payload)
    payload["truncated"] = True
    with pytest.raises(CPythonRightsAdmissionError, match="must be complete"):
        parse_pinned_tree_response(payload)


def test_canonical_build_rejects_partial_raw_tree_before_classification() -> None:
    row = _row("Lib/example.py", "value = 3\n")
    with pytest.raises(
        CPythonRightsAdmissionError,
        match="authenticated complete pinned tree authority required",
    ):
        build_admission([row], _tree(row), _policy())


def test_authenticated_tree_snapshot_isolated_from_caller_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("Lib/example.py", "value = 3\n")
    payload, authority = _synthetic_complete_authority(monkeypatch, row)
    blobs_before, evidence_before = rights._validated_tree_snapshot(authority)
    payload["tree"].clear()
    blobs_after, evidence_after = rights._validated_tree_snapshot(authority)
    assert blobs_before == blobs_after
    assert evidence_before == evidence_after
    assert blobs_after["Lib/example.py"] == row["git_blob_sha1"]


def test_authenticated_tree_omission_addition_and_substitution_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("Lib/example/module.py", "value = 3\n")
    policy = _policy()
    _, authority = _synthetic_complete_authority(
        monkeypatch,
        row,
        extra=[
            {
                "path": "Lib/example/LICENSE",
                "mode": "100644",
                "type": "blob",
                "sha": "1" * 40,
                "size": 7,
            }
        ],
    )
    snapshot, _ = rights._validated_tree_snapshot(authority)
    assert classify_row(row, snapshot, policy)["reason"] == "ANCESTOR_RIGHTS_MARKER_PRESENT"

    omitted = PinnedTreeAuthority(
        tuple(item for item in authority.entries if item[0] != "Lib/example/LICENSE")
    )
    with pytest.raises(CPythonRightsAdmissionError):
        rights._validated_tree_snapshot(omitted)

    added = PinnedTreeAuthority(
        authority.entries
        + (("Lib/unrelated.py", "100644", "blob", "3" * 40, 9),)
    )
    with pytest.raises(CPythonRightsAdmissionError):
        rights._validated_tree_snapshot(added)

    replaced_entries = list(authority.entries)
    target = next(
        i for i, item in enumerate(replaced_entries) if item[0] == row["source_path"]
    )
    path, mode, kind, _sha, size = replaced_entries[target]
    replaced_entries[target] = (path, mode, kind, "4" * 40, size)
    with pytest.raises(CPythonRightsAdmissionError):
        rights._validated_tree_snapshot(PinnedTreeAuthority(tuple(replaced_entries)))


def test_complete_tree_rejects_duplicate_path_and_bool_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("Lib/example.py", "value = 3\n")
    payload, _ = _synthetic_complete_authority(monkeypatch, row)
    duplicate = deepcopy(payload)
    duplicate["tree"].append(deepcopy(duplicate["tree"][-1]))
    monkeypatch.setattr(rights, "EXPECTED_TREE_ENTRY_COUNT", len(duplicate["tree"]))
    with pytest.raises(CPythonRightsAdmissionError, match="duplicate tree path"):
        parse_pinned_tree_response(duplicate)

    bad_size = deepcopy(payload)
    bad_size["tree"][-1]["size"] = True
    monkeypatch.setattr(rights, "EXPECTED_TREE_ENTRY_COUNT", len(bad_size["tree"]))
    with pytest.raises(CPythonRightsAdmissionError, match="tree object size invalid"):
        parse_pinned_tree_response(bad_size)


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



def test_pinned_tree_policy_uses_python_codepoint_order_digests() -> None:
    policy = _policy()
    assert rights.EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256 == (
        "fcd8a90eeb7ba5816e360f7c336b9fe8d6d56adebb51fbb4cecb13e71b6b3d1c"
    )
    assert rights.EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256 == (
        "85316aa5932aae00757d3ee9838201a17794400fdde656ac2a21085f5fcc1286"
    )
    assert EXPECTED_POLICY_IDENTITY_SHA256 == (
        "f4436690d6ea91b17fd0a5a7f110162bc051976eb6083b71c39c1ef274078346"
    )
    assert policy["scope"]["complete_tree_semantics_identity_sha256"] == (
        rights.EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256
    )
    assert policy["scope"]["complete_tree_blob_map_identity_sha256"] == (
        rights.EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256
    )
    assert policy["truth_boundary"]["authorized_optimized_target_exposure"] == 0


def test_complete_tree_snapshot_uses_python_codepoint_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uppercase = _row(".github/CODEOWNERS", "a = 1\\n")
    lowercase = _row(".github/actionlint.yaml", "b = 2\\n")
    payload, _ = _synthetic_complete_authority(monkeypatch, uppercase, lowercase)
    payload["tree"].reverse()
    authority = parse_pinned_tree_response(payload)
    paths = [entry[0] for entry in authority.entries]
    assert paths == sorted(paths)
    assert paths.index(".github/CODEOWNERS") < paths.index(".github/actionlint.yaml")


def test_exact_upstream_tree_python_order_replay_when_requested() -> None:
    """Optional physical Python replay: do not substitute synthetic fixtures."""
    if os.environ.get("TWELVE_SIX_CPYTHON_TREE_LIVE_REPLAY") != "1":
        pytest.skip("explicit immutable upstream tree replay not requested")
    authority = rights.fetch_pinned_tree_authority()
    snapshot, evidence = rights._validated_tree_snapshot(authority)
    assert snapshot["LICENSE"] == ROOT_LICENSE_BLOB_SHA1
    assert len(snapshot) == rights.EXPECTED_TREE_BLOB_COUNT
    assert evidence["semantics_identity_sha256"] == (
        rights.EXPECTED_TREE_SEMANTICS_IDENTITY_SHA256
    )
    assert evidence["blob_map_identity_sha256"] == (
        rights.EXPECTED_TREE_BLOB_MAP_IDENTITY_SHA256
    )
