from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.data import cluster_safe_split_guard as guard


def _sha(label: str) -> str:
    return guard._sha256_bytes(label.encode("utf-8"))


def _rehash(value: dict, key: str) -> None:
    value[key] = guard._sha256_obj_without(value, key)


def _handoff() -> dict:
    records = [
        {
            "record_id": "code-a",
            "source_id": "github:example/repo:file-a.py",
            "family": "github:example/repo",
            "modality": "code",
            "payload_sha256": _sha("code-a"),
            "payload_bytes": 100,
            "independence_cluster_identity_sha256": _sha("origin-repo"),
            "evaluation_reserved": False,
            "reserved_split": None,
        },
        {
            "record_id": "code-b",
            "source_id": "github:example/repo:file-b.py",
            "family": "github:example/repo",
            "modality": "code",
            "payload_sha256": _sha("code-b"),
            "payload_bytes": 120,
            "independence_cluster_identity_sha256": _sha("origin-repo"),
            "evaluation_reserved": False,
            "reserved_split": None,
        },
        {
            "record_id": "selection-uk",
            "source_id": "ua:reserved:selection",
            "family": "ua.reserved.selection",
            "modality": "uk",
            "payload_sha256": _sha("selection-uk"),
            "payload_bytes": 80,
            "independence_cluster_identity_sha256": _sha("origin-selection"),
            "evaluation_reserved": True,
            "reserved_split": "selection",
        },
    ]
    value = {
        "schema_version": guard.HANDOFF_SCHEMA,
        "decontamination_authority_identity_sha256": _sha("decontamination"),
        "decontamination_terminal": True,
        "reserved_evaluation_decontamination_complete": True,
        "record_count": len(records),
        "records": records,
        "raw_text_emitted": False,
        "final_test_payload_read": False,
        "authorized_training_exposure": 0,
    }
    _rehash(value, "handoff_identity_sha256")
    return value


def _manifest(handoff: dict) -> dict:
    value = {
        "schema_version": guard.SPLIT_MANIFEST_SCHEMA,
        "split_policy": guard.SPLIT_POLICY,
        "input_handoff_identity_sha256": handoff["handoff_identity_sha256"],
        "assignments": [
            {"record_id": "code-a", "split": "train"},
            {"record_id": "code-b", "split": "train"},
            {"record_id": "selection-uk", "split": "selection"},
        ],
    }
    _rehash(value, "split_manifest_identity_sha256")
    return value


def _verify(handoff: dict, manifest: dict) -> dict:
    return guard.verify_cluster_safe_split(
        handoff,
        manifest,
        expected_decontamination_authority_sha256=handoff[
            "decontamination_authority_identity_sha256"
        ],
        expected_handoff_identity_sha256=handoff["handoff_identity_sha256"],
        expected_split_manifest_identity_sha256=manifest[
            "split_manifest_identity_sha256"
        ],
    )


def test_accepts_complete_cluster_atomic_purpose_locked_split() -> None:
    handoff = _handoff()
    manifest = _manifest(handoff)
    proof = _verify(handoff, manifest)
    assert proof["record_count"] == 3
    assert proof["independence_cluster_count"] == 2
    assert proof["split_record_counts"] == {
        "final_test": 0,
        "selection": 1,
        "train": 2,
    }
    assert proof["split_payload_bytes"] == {
        "final_test": 0,
        "selection": 80,
        "train": 220,
    }
    assert proof["independence_clusters_cross_splits"] is False
    assert proof["evaluation_reservation_purpose_drift"] is False
    assert proof["authorized_training_exposure"] == 0
    assert proof["final_test_payload_read"] is False


def test_same_origin_siblings_cannot_cross_split_even_when_both_survive_dedup() -> None:
    handoff = _handoff()
    handoff["records"][1]["evaluation_reserved"] = True
    handoff["records"][1]["reserved_split"] = "selection"
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    manifest["input_handoff_identity_sha256"] = handoff["handoff_identity_sha256"]
    manifest["assignments"][1]["split"] = "selection"
    _rehash(manifest, "split_manifest_identity_sha256")
    with pytest.raises(guard.ClusterSafeSplitError, match="independence cluster crosses"):
        _verify(handoff, manifest)


def test_selection_reservation_cannot_be_reassigned_to_final_test() -> None:
    handoff = _handoff()
    manifest = _manifest(handoff)
    manifest["assignments"][2]["split"] = "final_test"
    _rehash(manifest, "split_manifest_identity_sha256")
    with pytest.raises(guard.ClusterSafeSplitError, match="wrong held-out purpose"):
        _verify(handoff, manifest)


def test_evaluation_reserved_record_cannot_enter_train() -> None:
    handoff = _handoff()
    manifest = _manifest(handoff)
    manifest["assignments"][2]["split"] = "train"
    _rehash(manifest, "split_manifest_identity_sha256")
    with pytest.raises(guard.ClusterSafeSplitError, match="wrong held-out purpose"):
        _verify(handoff, manifest)


def test_reserved_record_requires_exact_reserved_split_authority() -> None:
    handoff = _handoff()
    handoff["records"][2]["reserved_split"] = None
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    with pytest.raises(guard.ClusterSafeSplitError, match="reserved_split must be selection"):
        _verify(handoff, manifest)


def test_non_reserved_record_cannot_claim_held_out_reservation() -> None:
    handoff = _handoff()
    handoff["records"][0]["reserved_split"] = "selection"
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    with pytest.raises(guard.ClusterSafeSplitError, match="reserved_split must be null"):
        _verify(handoff, manifest)


def test_non_reserved_record_cannot_be_used_as_held_out_quota_repair() -> None:
    handoff = _handoff()
    handoff["records"][1]["independence_cluster_identity_sha256"] = _sha("origin-code-b")
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    manifest["input_handoff_identity_sha256"] = handoff["handoff_identity_sha256"]
    manifest["assignments"][1]["split"] = "selection"
    _rehash(manifest, "split_manifest_identity_sha256")
    with pytest.raises(guard.ClusterSafeSplitError, match="non-reserved record"):
        _verify(handoff, manifest)


def test_split_manifest_must_cover_exact_terminal_record_set() -> None:
    handoff = _handoff()
    manifest = _manifest(handoff)
    manifest["assignments"].pop()
    _rehash(manifest, "split_manifest_identity_sha256")
    with pytest.raises(guard.ClusterSafeSplitError, match="coverage"):
        _verify(handoff, manifest)


def test_self_consistent_handoff_substitution_rejected_by_external_identity() -> None:
    handoff = _handoff()
    expected = handoff["handoff_identity_sha256"]
    substitute = deepcopy(handoff)
    substitute["records"][0]["payload_bytes"] += 1
    _rehash(substitute, "handoff_identity_sha256")
    manifest = _manifest(substitute)
    with pytest.raises(guard.ClusterSafeSplitError, match="expected identity"):
        guard.verify_cluster_safe_split(
            substitute,
            manifest,
            expected_decontamination_authority_sha256=substitute[
                "decontamination_authority_identity_sha256"
            ],
            expected_handoff_identity_sha256=expected,
            expected_split_manifest_identity_sha256=manifest[
                "split_manifest_identity_sha256"
            ],
        )


def test_nonterminal_decontamination_cannot_create_split_proof() -> None:
    handoff = _handoff()
    handoff["decontamination_terminal"] = False
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    with pytest.raises(guard.ClusterSafeSplitError, match="nonterminal"):
        _verify(handoff, manifest)


def test_one_source_cannot_claim_multiple_independence_clusters() -> None:
    handoff = _handoff()
    handoff["records"][1]["source_id"] = handoff["records"][0]["source_id"]
    handoff["records"][1]["independence_cluster_identity_sha256"] = _sha("forged-origin")
    _rehash(handoff, "handoff_identity_sha256")
    manifest = _manifest(handoff)
    with pytest.raises(guard.ClusterSafeSplitError, match="one source_id maps"):
        _verify(handoff, manifest)
