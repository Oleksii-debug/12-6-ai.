"""D06 provenance checks binding learned-20M evaluation isolation to training data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_evaluation_firewall_provenance(evidence: Mapping[str, Any]) -> list[str]:
    """Fail closed unless the terminal firewall is bound to the exact launch corpus.

    The producer lanes own corpus/decontamination construction.  This verifier only
    consumes their immutable identities and prevents a clean firewall from a
    different corpus/split/packing graph from authorizing the learned-20M pilot.
    """

    firewall = evidence.get("evaluation_firewall")
    if not isinstance(firewall, Mapping) or firewall.get("terminal") is not True:
        return []

    corpus = evidence.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("terminal") is not True:
        return ["evaluation_firewall.training_corpus_not_terminal"]

    blockers: list[str] = []
    links = {
        "training_corpus_authority_identity": corpus.get("identity"),
        "training_corpus_identity": corpus.get("corpus_identity"),
        "training_split_identity": corpus.get("split_identity"),
        "training_packing_identity": corpus.get("packing_identity"),
        "record_graph_identity": corpus.get("record_graph_identity"),
    }
    for key, expected in links.items():
        if not _nonempty_text(expected):
            blockers.append(f"corpus.{key}_source_missing")
            continue
        observed = firewall.get(key)
        if not _nonempty_text(observed):
            blockers.append(f"evaluation_firewall.{key}_missing")
        elif observed != expected:
            blockers.append(f"evaluation_firewall.{key}_mismatch")

    if firewall.get("decontamination_before_split") is not True:
        blockers.append("evaluation_firewall.decontamination_before_split_not_proven")
    if firewall.get("decontamination_before_packing") is not True:
        blockers.append("evaluation_firewall.decontamination_before_packing_not_proven")
    if firewall.get("reserved_evaluation_excluded_before_split") is not True:
        blockers.append("evaluation_firewall.reserved_exclusion_before_split_not_proven")

    decontamination_identity = firewall.get("decontamination_identity")
    if not _nonempty_text(decontamination_identity):
        blockers.append("evaluation_firewall.decontamination_identity_missing")
    if firewall.get("decontaminated_record_graph_identity") != corpus.get(
        "record_graph_identity"
    ):
        blockers.append("evaluation_firewall.decontaminated_record_graph_identity_mismatch")

    return sorted(set(blockers))
