"""Plan9 preflight must bind physical S1/2 S10 S12 S13 S14 EVAL233, never grant."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import plan2_s15_plan9_handoff_preflight_v1 as preflight


def _parents() -> tuple[dict, dict, dict, dict, dict, dict]:
    source = "1" * 64
    rights = {
        "source_count": 5, "canonical_source_family_count": 3,
        "source_level_license_training_permission_evidenced": True,
        "s3_s9_physical_admission": False,
        "production_release_authorized": False,
        "three_family_source_sha256": source,
        "manifest_sha256": "a" * 64,
    }
    training = {
        "source_families_total": 3,
        "train_document_count": 4,
        "heldout_document_count": 11,
        "heldout_plaintext_materialized": False,
        "training_corpus_authorized": False,
        "physical_s9_admitted": False,
        "upstream_three_family_sha256": source,
        "s10_split_manifest_sha256": "2" * 64,
        "manifest_sha256": "b" * 64,
    }
    tokenizer = {
        "source_train_partition_sha256": "b" * 64,
        "source_s10_split_sha256": "2" * 64,
        "actual_vocab_size": 32768,
        "fitted_merge_count": 32508,
        "heldout_payloads_fitted": False,
        "production_train_source_admitted": False,
        "production_release_authorized": False,
        "manifest_sha256": "c" * 64,
    }
    data = {
        "source_cohort_manifest_sha256": source,
        "source_train_partition_sha256": "b" * 64,
        "source_s10_split_sha256": "2" * 64,
        "frozen_s12_candidate_sha256": "c" * 64,
        "train_document_count": 4,
        "heldout_document_count": 11,
        "target_count": 83000,
        "block_count": 3000,
        "heldout_payloads_exposed": False,
        "plan9_optimizer_permission": False,
        "production_release_authorized": False,
        "independent_s13_s14_readback_sha256": "g" * 64,
        "ordered_exposure_chain_sha256": "9" * 64,
        "manifest_sha256": "d" * 64,
    }
    evaluation = {
        "physical_three_family_sha256": source,
        "real_final_test_custody_verified": True,
        "real_final_decontamination_clean": True,
        "real_final_test_outcomes_read": False,
        "selection_payload_scanned": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "real_final_test_record_count": 16,
        "manifest_sha256": "e" * 64,
    }
    compat = {
        "source_fit_manifest_sha256": "c" * 64,
        "candidate_vocab_size": 32768,
        "production_32k_model_spec_compatible": True,
        "checkpoint_weight_reuse_approved": False,
        "plan9_optimizer_handoff_granted": False,
        "production_backend_binding_granted": False,
        "manifest_sha256": "f" * 64,
    }
    return rights, training, tokenizer, data, evaluation, compat


def test_plan9_evidence_handoff_is_verifiable_but_never_live(
    tmp_path: Path,
) -> None:
    inputs = _parents()
    proof = preflight.assemble(*inputs)
    assert proof["decision"] == "PLAN9_DATA_HANDOFF_PREFLIGHT_ONLY_NOT_PRODUCTION"
    assert proof["migration_contract"] == "migration-contract-baseline-v1"
    assert proof["source_family_count"] == 3
    assert proof["physical_training_document_count"] == 4
    assert proof["final_test_document_count"] == 16
    assert proof["frozen_candidate_vocab_size"] == 32768
    assert proof["physical_train_targets"] == 83000
    assert proof["validation_final_payloads_in_handoff"] is False
    assert proof["optimizer_effect_authorized"] is False
    assert proof["production_plan9_bindable"] is False
    assert proof["terminal_done"] is False
    assert len(proof["parent_manifests"]) == 7
    assert len(set(proof["parent_manifests"].values())) == 7
    assert len(proof["manifest_sha256"]) == 64
    result = preflight.stage_from_receipts(*inputs, destination=tmp_path / "out")
    assert result == proof
    assert preflight.stage_from_receipts(
        *inputs, destination=tmp_path / "out",
    ) == proof


@pytest.mark.parametrize("name", [
    "rights", "split", "tokenizer", "exposure", "eval", "checkpoint", "source",
])
def test_unqualified_plan9_handoff_denied(name: str) -> None:
    items = list(_parents())
    if name == "rights":
        items[0]["s3_s9_physical_admission"] = True
    elif name == "split":
        items[1]["heldout_plaintext_materialized"] = True
    elif name == "tokenizer":
        items[2]["actual_vocab_size"] = 260
    elif name == "exposure":
        items[3]["heldout_payloads_exposed"] = True
    elif name == "eval":
        items[4]["real_final_decontamination_clean"] = False
    elif name == "checkpoint":
        items[5]["checkpoint_weight_reuse_approved"] = True
    else:
        items[3]["source_cohort_manifest_sha256"] = "8" * 64
    with pytest.raises(preflight.Plan9HandoffDenied):
        preflight.assemble(*items)


def test_plan9_preflight_immutable_output_denied_after_tamper(
    tmp_path: Path,
) -> None:
    inputs = _parents()
    destination = tmp_path / "plan9"
    preflight.stage_from_receipts(*inputs, destination=destination)
    (destination / preflight.OUTPUT).write_bytes(
        b'{"production_plan9_bindable":true}\n'
    )
    with pytest.raises(preflight.Plan9HandoffDenied, match="immutable"):
        preflight.stage_from_receipts(*inputs, destination=destination)


def test_plan9_preflight_symlink_blocks_publication(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "target"
    destination.mkdir()
    link = tmp_path / "link"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(preflight.Plan9HandoffDenied, match="symlink"):
        preflight.stage_from_receipts(*_parents(), destination=link)
