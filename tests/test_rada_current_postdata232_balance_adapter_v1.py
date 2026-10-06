from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import tools.adapt_d03_rada_current_postdata232_to_balance_v1 as target


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _row(
    record_id: str,
    family: str,
    modality: str,
    payload_bytes: int,
    *,
    source_id: str | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_id": source_id or f"source:{record_id}",
        "family": family,
        "modality": modality,
        "payload_sha256": _sha(record_id.encode("utf-8")),
        "payload_bytes": payload_bytes,
    }


def _inventory(
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    selected = rows or [
        _row("r-uk", "ua.rada.open-data.laws-texts", "uk", 1200),
        _row("r-en", "en.standardebooks.manual", "en", 900),
        _row("r-code", "github:encode/httpx", "code", 700),
    ]
    selected = sorted(copy.deepcopy(selected), key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in selected
    ]
    return {
        "schema_version": target.INVENTORY_SCHEMA,
        "record_count": len(selected),
        "total_payload_bytes": sum(int(row["payload_bytes"]) for row in selected),
        "record_inventory_digest_sha256": _sha(target.canonical(selected)),
        "payload_inventory_digest_sha256": _sha(
            target.canonical(payload_projection)
        ),
        "records": selected,
    }


def _receipt(
    inventory: dict[str, object],
    *,
    inventory_file_sha256: str,
    record_payload_jsonl_sha256: str,
) -> dict[str, object]:
    payload_set_projection = sorted(
        (
            {
                "payload_sha256": row["payload_sha256"],
                "payload_bytes": row["payload_bytes"],
            }
            for row in inventory["records"]
        ),
        key=lambda row: (row["payload_sha256"], row["payload_bytes"]),
    )
    payload_set_identity = _sha(target.canonical(payload_set_projection))
    core: dict[str, object] = {
        "schema_version": target.PARENT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": target.STACK_BASE_HEAD,
        "parent": {
            "execution_head_sha": target.PARENT_DATA232_HEAD,
            "product_data232_head_sha": target.PRODUCT_DATA232_HEAD,
            "artifact_id": 1,
            "artifact_zip_sha256": "1" * 64,
            "inventory_identity_sha256": "2" * 64,
            "training_handoff_identity_sha256": "3" * 64,
            "data232_report_sha256": "4" * 64,
            "data232_execution_identity_sha256": "5" * 64,
            "data232_result_identity_sha256": "6" * 64,
            "data232_two_clean_proof_identity_sha256": "7" * 64,
            "full_selection_projection_sha256":
                target.EXPECTED_FULL_SELECTION_SHA256,
            "current_rada_slice_authority_sha256":
                target.EXPECTED_RADA_SLICE_SHA256,
            "two_fresh_data232_processes_byte_identical": True,
        },
        "g05": {
            "input_rows_sha256": "8" * 64,
            "execution_identity_sha256": "9" * 64,
            "counts": {},
            "bytes": {},
            "partial_materialization": {},
        },
        "g06": {
            "input_rows_sha256": "a" * 64,
            "execution_identity_sha256": "b" * 64,
            "detector_counts": {},
            "materialization": {},
            "payload_delta_from_g05_retained_bytes": 0,
            "exact_payload_collision_free": True,
            "unique_payload_count": inventory["record_count"],
            "payload_set_identity_sha256": payload_set_identity,
        },
        "durable_artifacts": {
            "g05_authority_file_sha256": "c" * 64,
            "g06_authority_file_sha256": "d" * 64,
            "survivor_inventory_file_sha256": inventory_file_sha256,
            "completion_marker": "POST_G05_G06_EVIDENCE_WRITTEN_LAST",
        },
        "survivor_inventory": {
            "record_payload_jsonl_sha256": record_payload_jsonl_sha256,
            "record_count": inventory["record_count"],
            "total_payload_bytes": inventory["total_payload_bytes"],
            "record_inventory_digest_sha256":
                inventory["record_inventory_digest_sha256"],
            "payload_inventory_digest_sha256":
                inventory["payload_inventory_digest_sha256"],
        },
        "counts": {
            "input_training_records": 10,
            "data232_excluded_records": 1,
            "post_data232_records": 9,
            "post_g05_records": inventory["record_count"],
            "post_g06_records": inventory["record_count"],
            "post_g06_source_objects": len(
                {row["source_id"] for row in inventory["records"]}
            ),
            "post_g06_payload_bytes": inventory["total_payload_bytes"],
        },
        "authority_blobs": {
            "synthetic": "e" * 40,
        },
        "content_boundary": dict(target.EXPECTED_CONTENT_BOUNDARY),
        "truth_boundary": dict(target.EXPECTED_TRUTH_BOUNDARY),
        "next_gate": "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST",
    }
    return {
        **core,
        "evidence_identity_sha256": _sha(target.canonical(core)),
    }


def _write_inputs(
    tmp_path: Path,
    rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, dict[str, object], dict[str, object], str]:
    inventory = _inventory(rows)
    inventory_raw = target.canonical_line(inventory)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(inventory_raw)

    jsonl_sha = _sha(b"ephemeral canonical survivor jsonl\n")
    receipt = _receipt(
        inventory,
        inventory_file_sha256=_sha(inventory_raw),
        record_payload_jsonl_sha256=jsonl_sha,
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_bytes(target.canonical_line(receipt))
    return evidence_path, inventory_path, receipt, inventory, jsonl_sha


def _args(
    tmp_path: Path,
    evidence_path: Path,
    inventory_path: Path,
    receipt: dict[str, object],
    inventory: dict[str, object],
    jsonl_sha: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        evidence=evidence_path,
        inventory=inventory_path,
        expected_parent_execution_head=target.STACK_BASE_HEAD,
        expected_evidence_file_sha256=_sha(evidence_path.read_bytes()),
        expected_evidence_identity_sha256=receipt[
            "evidence_identity_sha256"
        ],
        expected_inventory_file_sha256=_sha(inventory_path.read_bytes()),
        expected_record_payload_jsonl_sha256=jsonl_sha,
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        expected_record_count=inventory["record_count"],
        expected_total_payload_bytes=inventory["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in inventory["records"]}
        ),
        output=tmp_path / "family-vector.json",
    )


def test_exact_synthetic_receipt_builds_canonical_family_vector(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    result = target.execute(_args(tmp_path, *paths))
    vector = target.load_json_bytes(
        (tmp_path / "family-vector.json").read_bytes(),
        "vector",
    )

    verify, _, _ = target.load_canonical_bridge()
    assert verify(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]
    assert result["record_count"] == 3
    assert result["total_payload_bytes"] == 2800
    assert vector["stratum_capacity_bytes"] == {
        "code": 700,
        "en": 900,
        "uk": 1200,
    }
    assert vector["authorized_optimized_target_exposure"] == 0
    assert vector["tokenizer_fit_authorized"] is False
    assert vector["model_training_authorized"] is False
    serialized = (tmp_path / "family-vector.json").read_text("utf-8")
    assert "normalized_payload" not in serialized


def test_canonical_dependency_preflight_accepts_pinned_branch() -> None:
    assert target.verify_canonical_dependency_blobs() == dict(
        sorted(target.EXPECTED_CANONICAL_BLOBS.items())
    )


def test_canonical_dependency_preflight_rejects_worktree_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = target._git

    def fake_git(*args: str) -> str:
        if args[:1] == ("hash-object",):
            return "f" * 40
        return original(*args)

    monkeypatch.setattr(target, "_git", fake_git)
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="working-tree blob drift",
    ):
        target.verify_canonical_dependency_blobs()


def test_parent_truth_widening_fails_even_when_receipt_is_resealed(
    tmp_path: Path,
) -> None:
    evidence_path, inventory_path, receipt, inventory, jsonl_sha = _write_inputs(
        tmp_path
    )
    receipt["truth_boundary"]["training_executed"] = True
    core = dict(receipt)
    core.pop("evidence_identity_sha256")
    receipt["evidence_identity_sha256"] = _sha(target.canonical(core))
    evidence_path.write_bytes(target.canonical_line(receipt))
    args = _args(
        tmp_path,
        evidence_path,
        inventory_path,
        receipt,
        inventory,
        jsonl_sha,
    )
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="zero-credit truth boundary drift",
    ):
        target.execute(args)


def test_parent_head_drift_fails_even_when_receipt_is_resealed(
    tmp_path: Path,
) -> None:
    evidence_path, inventory_path, receipt, inventory, jsonl_sha = _write_inputs(
        tmp_path
    )
    receipt["execution_head_sha"] = "f" * 40
    core = dict(receipt)
    core.pop("evidence_identity_sha256")
    receipt["evidence_identity_sha256"] = _sha(target.canonical(core))
    evidence_path.write_bytes(target.canonical_line(receipt))
    args = _args(
        tmp_path,
        evidence_path,
        inventory_path,
        receipt,
        inventory,
        jsonl_sha,
    )
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="execution head drift",
    ):
        target.execute(args)


def test_coherently_resealed_inventory_still_fails_external_roots(
    tmp_path: Path,
) -> None:
    evidence_path, inventory_path, receipt, inventory, jsonl_sha = _write_inputs(
        tmp_path
    )
    original_args = _args(
        tmp_path,
        evidence_path,
        inventory_path,
        receipt,
        inventory,
        jsonl_sha,
    )

    mutated = copy.deepcopy(inventory)
    mutated["records"][0]["payload_bytes"] += 1
    mutated = _inventory(mutated["records"])
    mutated_raw = target.canonical_line(mutated)
    inventory_path.write_bytes(mutated_raw)

    resealed = _receipt(
        mutated,
        inventory_file_sha256=_sha(mutated_raw),
        record_payload_jsonl_sha256=jsonl_sha,
    )
    evidence_path.write_bytes(target.canonical_line(resealed))

    original_args.expected_evidence_file_sha256 = _sha(
        evidence_path.read_bytes()
    )
    original_args.expected_evidence_identity_sha256 = resealed[
        "evidence_identity_sha256"
    ]
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="inventory file SHA-256 drift|receipt .* drift",
    ):
        target.execute(original_args)


def test_unknown_family_fails_before_vector_publication(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        [_row("unknown", "unknown.family", "en", 100)],
    )
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="untrusted survivor family",
    ):
        target.execute(_args(tmp_path, *paths))
    assert not (tmp_path / "family-vector.json").exists()


def test_family_modality_drift_fails_closed(tmp_path: Path) -> None:
    paths = _write_inputs(
        tmp_path,
        [_row("bad", "github:encode/httpx", "en", 100)],
    )
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="code family modality drift",
    ):
        target.execute(_args(tmp_path, *paths))


def test_inventory_raw_text_key_is_rejected(tmp_path: Path) -> None:
    inventory = _inventory()
    inventory["records"][0]["normalized_payload"] = "secret"
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_bytes(target.canonical_line(inventory))
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="fields drift",
    ):
        target.verify_inventory(
            inventory,
            expected_record_count=3,
            expected_total_payload_bytes=2800,
            expected_source_object_count=3,
            expected_record_inventory_digest_sha256="0" * 64,
            expected_payload_inventory_digest_sha256="0" * 64,
            expected_payload_set_identity_sha256="0" * 64,
        )


def test_duplicate_json_keys_fail_closed() -> None:
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="strict JSON decode failed",
    ):
        target.load_json_bytes(b'{"a":1,"a":2}', "duplicate")


def test_nonfinite_json_fails_closed() -> None:
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="strict JSON decode failed",
    ):
        target.load_json_bytes(b'{"a":NaN}', "nan")


def test_immutable_output_allows_identical_resume_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "out.json"
    target.write_immutable(path, b"same\n")
    target.write_immutable(path, b"same\n")
    assert path.read_bytes() == b"same\n"
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="divergent output overwrite",
    ):
        target.write_immutable(path, b"different\n")


def test_record_payload_jsonl_root_is_carried_without_raw_payload(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    target.execute(_args(tmp_path, *paths))
    vector = target.load_json_bytes(
        (tmp_path / "family-vector.json").read_bytes(),
        "vector",
    )
    assert vector["record_payload_jsonl_sha256"] == paths[-1]
    assert set(vector["families"][0]) == {
        "stratum",
        "family",
        "record_count",
        "capacity_bytes",
    }



def test_future_terminal_parent_head_can_be_bound_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        target,
        "verify_parent_execution_ancestry",
        lambda value: value,
    )
    evidence_path, inventory_path, receipt, inventory, jsonl_sha = _write_inputs(
        tmp_path
    )
    terminal_head = "f" * 40
    receipt["execution_head_sha"] = terminal_head
    core = dict(receipt)
    core.pop("evidence_identity_sha256")
    receipt["evidence_identity_sha256"] = _sha(target.canonical(core))
    evidence_path.write_bytes(target.canonical_line(receipt))

    args = _args(
        tmp_path,
        evidence_path,
        inventory_path,
        receipt,
        inventory,
        jsonl_sha,
    )
    args.expected_parent_execution_head = terminal_head
    result = target.execute(args)
    vector = target.load_json_bytes(
        (tmp_path / "family-vector.json").read_bytes(),
        "vector",
    )

    assert vector["materialization_execution_head_sha"] == terminal_head
    assert vector["source_git_sha"] == terminal_head
    assert result["family_vector_identity_sha256"] == vector[
        "family_vector_identity_sha256"
    ]



def test_stack_base_is_valid_current_parent_ancestor() -> None:
    assert (
        target.verify_parent_execution_ancestry(target.STACK_BASE_HEAD)
        == target.STACK_BASE_HEAD
    )


def test_nonexistent_terminal_parent_head_is_rejected() -> None:
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="outside the stacked",
    ):
        target.verify_parent_execution_ancestry("f" * 40)

def test_canonical_bridge_discards_in_memory_trusted_family_mutation() -> None:
    _, trusted, _ = target.load_canonical_bridge()
    original_count = len(trusted)
    trusted["adversarial.in-memory.family"] = {
        "family": "adversarial.in-memory.family",
        "source_family_identity_sha256": "0" * 64,
        "language": "en",
        "modalities": ["text"],
        "stratum": "en",
    }

    _, refreshed, _ = target.load_canonical_bridge()

    assert "adversarial.in-memory.family" not in refreshed
    assert len(refreshed) == original_count


def test_canonical_bridge_rejects_alternate_module_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = target.importlib.import_module
    fake = types.SimpleNamespace(
        __file__=str(tmp_path / "trusted_family_authority_v1.py"),
        TRUSTED_FAMILY_SEMANTICS={},
        trusted_family_authority_root_sha256=lambda _families: "0" * 64,
    )

    def adversarial_import(name: str):
        if name == "twelve_six.data.trusted_family_authority_v1":
            return fake
        return real_import(name)

    monkeypatch.setattr(target.importlib, "import_module", adversarial_import)
    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="canonical module provenance drift",
    ):
        target.load_canonical_bridge()


def test_canonical_bridge_rejects_preloaded_foreign_top_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign = types.SimpleNamespace(
        __file__=str(tmp_path / "__init__.py"),
        __path__=[str(tmp_path)],
    )
    monkeypatch.setitem(sys.modules, "twelve_six", foreign)

    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="canonical twelve_six package provenance drift",
    ):
        target.load_canonical_bridge()

def test_inventory_rejects_post_g06_duplicate_payload_hash() -> None:
    payload = "a" * 64
    rows = [
        {
            "record_id": "r-a",
            "source_id": "s-a",
            "family": "ua.rada.open-data.laws-texts",
            "modality": "uk",
            "payload_sha256": payload,
            "payload_bytes": 10,
        },
        {
            "record_id": "r-b",
            "source_id": "s-b",
            "family": "ua.rada.open-data.laws-texts",
            "modality": "uk",
            "payload_sha256": payload,
            "payload_bytes": 11,
        },
    ]
    inventory = {
        "schema_version": target.INVENTORY_SCHEMA,
        "record_count": 2,
        "total_payload_bytes": 21,
        "record_inventory_digest_sha256": "0" * 64,
        "payload_inventory_digest_sha256": "1" * 64,
        "records": rows,
    }

    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="post-G06 inventory exact payload collision",
    ):
        target.verify_inventory(
            inventory,
            expected_record_count=2,
            expected_total_payload_bytes=21,
            expected_source_object_count=2,
            expected_record_inventory_digest_sha256="2" * 64,
            expected_payload_inventory_digest_sha256="3" * 64,
            expected_payload_set_identity_sha256="4" * 64,
        )


def test_canonical_bridge_rejects_foreign_data_namespace_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target.load_canonical_bridge()
    foreign_data = types.SimpleNamespace(__path__=[str(tmp_path)])
    monkeypatch.setitem(sys.modules, "twelve_six.data", foreign_data)

    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="canonical twelve_six.data package search path drift",
    ):
        target.load_canonical_bridge()

def test_inventory_cross_binds_parent_payload_set_identity() -> None:
    inventory = _inventory()
    rows = inventory["records"]
    payload_set_projection = sorted(
        (
            {
                "payload_sha256": row["payload_sha256"],
                "payload_bytes": row["payload_bytes"],
            }
            for row in rows
        ),
        key=lambda row: (row["payload_sha256"], row["payload_bytes"]),
    )
    actual_payload_set_identity = _sha(
        target.canonical(payload_set_projection)
    )

    verified = target.verify_inventory(
        inventory,
        expected_record_count=inventory["record_count"],
        expected_total_payload_bytes=inventory["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in rows}
        ),
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        expected_payload_set_identity_sha256=actual_payload_set_identity,
    )
    assert len(verified) == inventory["record_count"]

    with pytest.raises(
        target.CurrentRadaBalanceAdapterError,
        match="post-G06 payload-set identity drift",
    ):
        target.verify_inventory(
            inventory,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=len(
                {row["source_id"] for row in rows}
            ),
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            expected_payload_set_identity_sha256="f" * 64,
        )



def test_matrix_successor_exact_roots_are_pinned() -> None:
    assert target.STACK_BASE_HEAD == "51758daf703f58207cfca9ef4e64fc8d0cd203f0"
    assert target.PARENT_DATA232_HEAD == "a7982bdfd1650b062808024856e13b1d8916a634"
