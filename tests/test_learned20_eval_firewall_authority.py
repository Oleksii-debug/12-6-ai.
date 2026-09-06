from __future__ import annotations

from twelve_six.learned20_eval_firewall_authority import (
    validate_evaluation_firewall_provenance,
)


def _evidence() -> dict:
    corpus = {
        "terminal": True,
        "identity": "corpus-authority-v1",
        "corpus_identity": "corpus-v1",
        "split_identity": "split-v1",
        "packing_identity": "packing-v1",
        "record_graph_identity": "record-graph-v1",
    }
    firewall = {
        "terminal": True,
        "identity": "firewall-v1",
        "training_corpus_authority_identity": corpus["identity"],
        "training_corpus_identity": corpus["corpus_identity"],
        "training_split_identity": corpus["split_identity"],
        "training_packing_identity": corpus["packing_identity"],
        "record_graph_identity": corpus["record_graph_identity"],
        "decontamination_identity": "decontam-v1",
        "decontaminated_record_graph_identity": corpus["record_graph_identity"],
        "decontamination_before_split": True,
        "decontamination_before_packing": True,
        "reserved_evaluation_excluded_before_split": True,
    }
    return {"corpus": corpus, "evaluation_firewall": firewall}


def test_terminal_firewall_can_bind_exact_training_graph() -> None:
    assert validate_evaluation_firewall_provenance(_evidence()) == []


def test_clean_firewall_from_other_corpus_cannot_authorize_pilot() -> None:
    evidence = _evidence()
    evidence["evaluation_firewall"]["training_corpus_identity"] = "other-corpus"
    evidence["evaluation_firewall"]["training_split_identity"] = "other-split"
    blockers = validate_evaluation_firewall_provenance(evidence)
    assert "evaluation_firewall.training_corpus_identity_mismatch" in blockers
    assert "evaluation_firewall.training_split_identity_mismatch" in blockers


def test_record_graph_and_pre_split_order_are_mandatory() -> None:
    evidence = _evidence()
    del evidence["corpus"]["record_graph_identity"]
    evidence["evaluation_firewall"]["decontamination_before_split"] = False
    evidence["evaluation_firewall"]["reserved_evaluation_excluded_before_split"] = False
    blockers = validate_evaluation_firewall_provenance(evidence)
    assert "corpus.record_graph_identity_source_missing" in blockers
    assert "evaluation_firewall.decontamination_before_split_not_proven" in blockers
    assert "evaluation_firewall.reserved_exclusion_before_split_not_proven" in blockers


def test_terminal_firewall_requires_terminal_corpus() -> None:
    evidence = _evidence()
    evidence["corpus"]["terminal"] = False
    assert validate_evaluation_firewall_provenance(evidence) == [
        "evaluation_firewall.training_corpus_not_terminal"
    ]
