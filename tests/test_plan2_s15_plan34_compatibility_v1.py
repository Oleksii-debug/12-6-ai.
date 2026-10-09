"""Plan3/4 checks must refuse an S12 fixture-vocab production handoff."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import plan2_s15_plan34_compatibility_v1 as compat
from tools import plan2_tokenizer_fit_freeze_v1 as frozen


def _fitted() -> dict:
    identity = frozen.FrozenBPE([]).identity.to_dict()
    return {
        "decision": "THREE_FAMILY_TRAIN_ONLY_BPE_CANDIDATE_NOT_PRODUCTION",
        "manifest_sha256": "a" * 64,
        "source_train_document_count": 1,
        "heldout_payloads_fitted": False,
        "production_target_vocab_frozen": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "target_vocab_size": 32768,
        "candidate_vocab_size": identity["vocab_size"],
        "tokenizer_identity": identity,
    }


def test_canonical_migration_guard_refuses_32k_production() -> None:
    proof = compat.assess(_fitted())
    assert proof["decision"] == (
        "PLAN34_FRESH_CANDIDATE_COMPATIBLE_PRODUCTION_32K_REFUSED"
    )
    assert proof["candidate_vocab_size"] == 260
    assert proof["required_production_vocab_size"] == 32768
    assert proof["candidate_fresh_random_initialization_migration_compatible"] is True
    assert proof["checkpoint_weight_reuse_approved"] is False
    assert proof["production_32k_model_spec_compatible"] is False
    assert proof["production_backend_binding_granted"] is False
    assert proof["plan9_optimizer_handoff_granted"] is False
    assert proof["terminal_done"] is False
    assert len(proof["candidate_migration_decision_sha256"]) == 64
    assert compat.assess(_fitted()) == proof


@pytest.mark.parametrize("change", [
    "fit_grant", "corpus_grant", "promote_32k",
    "holdout_payloads_fitted", "candidate_vocab_identity",
])
def test_false_production_migration_never_passes(change: str) -> None:
    value = _fitted()
    if change == "fit_grant":
        value["tokenizer_fit_authorized"] = True
    elif change == "corpus_grant":
        value["training_corpus_authorized"] = True
    elif change == "promote_32k":
        value["production_target_vocab_frozen"] = True
    elif change == "holdout_payloads_fitted":
        value["heldout_payloads_fitted"] = True
    else:
        value["candidate_vocab_size"] = 32767
    with pytest.raises(compat.Plan34CompatibilityDenied):
        compat.assess(value)


def test_compatibility_publication_tampering_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compat, "inspect", lambda _: compat.assess(_fitted()))
    out = tmp_path / "out"
    report = compat.stage(tmp_path, out)
    assert report["terminal_done"] is False
    (out / compat.OUTPUT).write_bytes(
        b'{"production_backend_binding_granted":true}\n'
    )
    with pytest.raises(compat.Plan34CompatibilityDenied, match="immutable"):
        compat.stage(tmp_path, out)


def test_symlinked_compatibility_publication_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "actual"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(compat, "inspect", lambda _: pytest.fail("source read"))
    with pytest.raises(compat.Plan34CompatibilityDenied, match="symlink"):
        compat.stage(tmp_path, link)
