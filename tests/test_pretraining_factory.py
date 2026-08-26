from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.external_sources import (
    ExternalSourceSpec,
    ReservedSetSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
    build_reserved_fingerprint_registry,
)
from twelve_six.data.pretraining_factory import (
    FactoryPlan,
    PretrainingFactoryError,
    build_token_targets,
    finalize_jsonl_and_tokenizer_input,
    prepare_exact_stage,
    run_local_fixture_near_dedup,
    validate_datatrove_runtime,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path, *, approved: bool = True):
    normal = (
        "A sufficiently long English synthetic sentence provides deterministic data factory "
        "mechanics while carrying no external corpus rights claim."
    )
    contaminated = (
        "A reserved benchmark sentence is deliberately injected so exact normalized content "
        "decontamination can be proven by this controlled regression fixture."
    )
    rows = [
        {"document_id": "a", "text": normal, "language_hint": "en"},
        {"document_id": "b", "text": normal, "language_hint": "en"},
        {
            "document_id": "c",
            "text": (
                "Another distinct English synthetic document contains enough alphabetic words "
                "to remain eligible after local quality filtering for the pipeline test."
            ),
            "language_hint": "en",
        },
        {
            "document_id": "d",
            "text": (
                "A third distinct English synthetic document contains many words and supports "
                "a nonempty deterministic train validation split in the regression test."
            ),
            "language_hint": "en",
        },
        {"document_id": "reserved", "text": contaminated, "language_hint": "en"},
        {
            "document_id": "pii",
            "text": (
                "This otherwise acceptable English sentence includes test@example.com and must "
                "be rejected by the explicit PII policy hook before deduplication."
            ),
            "language_hint": "en",
        },
    ]
    raw = tmp_path / "raw.jsonl"
    raw_bytes = b"".join(_canonical(row) for row in rows)
    raw.write_bytes(raw_bytes)
    source_registry = build_external_source_registry(
        [
            ExternalSourceSpec(
                source_id="fixture",
                source_version="v1",
                provider="12-6-test-suite",
                source_url="https://example.invalid/fixture",
                source_kind="jsonl_text_v1",
                purpose="pretraining",
                synthetic=True,
                benchmark_material=False,
                held_out=False,
                snapshot=SnapshotSpec(
                    uri=raw.as_uri(),
                    sha256=_sha(raw_bytes),
                    size_bytes=len(raw_bytes),
                    retrieved_at="2026-08-25T00:00:00Z",
                    upstream_version="v1",
                    retrieval_method="test-fixture",
                ),
                rights=RightsDecision(
                    status="APPROVED_FOR_TRAINING" if approved else "REVIEW_REQUIRED",
                    license_id="CC0-1.0",
                    terms_url="https://example.invalid/fixture-terms",
                    allows_model_training=approved,
                    allows_derivatives=True,
                    allows_redistribution=True,
                    policy_ref="test-only-controlled-fixture",
                    reviewed_at="2026-08-25T00:00:00Z",
                    reviewer_ref="test-suite",
                ),
            )
        ]
    )
    registry_id = source_registry["registry_identity_sha256"]
    receipt = {
        "schema_version": "12-6.source-retrieval-receipt.v1",
        "source_registry_identity_sha256": registry_id,
        "source_id": "fixture",
        "source_version": "v1",
        "destination_uri": raw.as_uri(),
        "expected_sha256": _sha(raw_bytes),
        "verified_sha256": _sha(raw_bytes),
        "verified_size_bytes": len(raw_bytes),
        "verification": "PASS",
        "training_eligibility_evaluated": False,
    }
    inventory_core = {
        "schema_version": "12-6.source-retrieval-inventory.v1",
        "source_registry_identity_sha256": registry_id,
        "receipts": [receipt],
        "rights_semantics": "INVENTORY_IS_NOT_TRAINING_APPROVAL",
    }
    inventory = {
        **inventory_core,
        "inventory_sha256": _sha(_canonical(inventory_core)),
    }
    reserved = build_reserved_fingerprint_registry(
        [
            ReservedSetSpec(
                set_id="fixture-reserved",
                version="v1",
                source_id="benchmark-fixture",
                purpose="benchmark",
                normalized_sha256=(_sha(contaminated.encode("utf-8")),),
            )
        ]
    )
    output = tmp_path / "out"
    output.mkdir()
    plan = FactoryPlan(
        source_registry_sha256=registry_id,
        retrieval_inventory_sha256=inventory["inventory_sha256"],
        reserved_registry_sha256=reserved["registry_identity_sha256"],
        output_uri=output.as_uri(),
        validation_per_10k=5_000,
        shard_count=2,
    )
    return plan, source_registry, inventory, reserved


def test_token_targets_do_not_freeze_tokenizer() -> None:
    result = build_token_targets(10_000_000)
    assert result["targets_in_selected_experiment_tokenizer_tokens"]["scratch_baseline"] == 200_000_000
    assert result["canonical_tokenizer_frozen"] is False


def test_factory_filters_dedups_decontaminates_splits_and_resumes(tmp_path: Path) -> None:
    plan, registry, inventory, reserved = _fixture(tmp_path)
    exact = prepare_exact_stage(plan, registry, inventory, reserved)
    assert exact["counters"]["exact_duplicates_removed"] == 1
    assert exact["counters"]["benchmark_contamination_rejected"] == 1
    assert exact["counters"]["rejected_pii_email"] == 1
    assert exact["record_count"] == 3
    near = run_local_fixture_near_dedup(plan, exact["records_uri"])
    assert near["production_near_dedup_executed"] is False
    final = finalize_jsonl_and_tokenizer_input(
        plan,
        near["output_uri"],
        near_dedup_evidence=near,
    )
    assert final["documents"] == near["output_records"]
    assert final["split_documents"]["train"] > 0
    assert final["split_documents"]["validation"] > 0
    assert final["tokenizer_input"]["canonical_tokenizer_selected"] is False
    assert prepare_exact_stage(plan, registry, inventory, reserved)["resumed"] is True
    assert finalize_jsonl_and_tokenizer_input(
        plan,
        near["output_uri"],
        near_dedup_evidence=near,
    )["resumed"] is True


def test_rights_gate_fails_before_processing(tmp_path: Path) -> None:
    plan, registry, inventory, reserved = _fixture(tmp_path, approved=False)
    with pytest.raises(ValueError, match="rights are not approved"):
        prepare_exact_stage(plan, registry, inventory, reserved)


def test_plan_rejects_reserved_registry_identity_drift(tmp_path: Path) -> None:
    plan, registry, inventory, reserved = _fixture(tmp_path)
    reserved["registry_identity_sha256"] = _sha(b"different")
    with pytest.raises(ValueError, match="registry identity mismatch"):
        prepare_exact_stage(plan, registry, inventory, reserved)


def test_datatrove_backend_fails_closed_without_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import twelve_six.data.pretraining_factory as factory

    def missing(_: str) -> str:
        raise factory.PackageNotFoundError

    monkeypatch.setattr(factory, "version", missing)
    with pytest.raises(RuntimeError, match="DataTrove optional runtime"):
        validate_datatrove_runtime()


def test_remote_output_uri_is_plannable_but_not_local_executable(tmp_path: Path) -> None:
    plan, registry, inventory, reserved = _fixture(tmp_path)
    remote = FactoryPlan(
        source_registry_sha256=plan.source_registry_sha256,
        retrieval_inventory_sha256=plan.retrieval_inventory_sha256,
        reserved_registry_sha256=plan.reserved_registry_sha256,
        output_uri="s3://twelve-six-corpus/example",
    )
    assert remote.manifest()["output_uri"].startswith("s3://")
    with pytest.raises(PretrainingFactoryError, match="LOCAL_FREE"):
        prepare_exact_stage(remote, registry, inventory, reserved)
