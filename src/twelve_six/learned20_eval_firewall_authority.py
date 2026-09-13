"""D06 provenance checks binding learned-20M evaluation isolation to training data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_HEX = frozenset("0123456789abcdef")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in _HEX for character in value)
    )


def validate_evaluation_firewall_provenance(evidence: Mapping[str, Any]) -> list[str]:
    """Fail closed unless the terminal firewall is bound to the exact launch corpus.

    Producer lanes own corpus/decontamination construction. This verifier consumes
    their immutable identities and the canonical DATA-526 record/payload inventory
    digests so a clean firewall from a different corpus/split/packing graph cannot
    authorize the learned-20M pilot.
    """

    firewall = evidence.get("evaluation_firewall")
    if not isinstance(firewall, Mapping) or firewall.get("terminal") is not True:
        return []

    corpus = evidence.get("corpus")
    if not isinstance(corpus, Mapping) or corpus.get("terminal") is not True:
        return ["evaluation_firewall.training_corpus_not_terminal"]

    blockers: list[str] = []
    identity_links = {
        "training_corpus_authority_identity": corpus.get("identity"),
        "training_corpus_identity": corpus.get("corpus_identity"),
        "training_split_identity": corpus.get("split_identity"),
        "training_packing_identity": corpus.get("packing_identity"),
    }
    for key, expected in identity_links.items():
        if not _nonempty_text(expected):
            blockers.append(f"corpus.{key}_source_missing")
            continue
        observed = firewall.get(key)
        if not _nonempty_text(observed):
            blockers.append(f"evaluation_firewall.{key}_missing")
        elif observed != expected:
            blockers.append(f"evaluation_firewall.{key}_mismatch")

    digest_links = {
        "record_inventory_digest_sha256": corpus.get("record_inventory_digest_sha256"),
        "payload_inventory_digest_sha256": corpus.get("payload_inventory_digest_sha256"),
    }
    for key, expected in digest_links.items():
        if not _sha256_text(expected):
            blockers.append(f"corpus.{key}_source_invalid")
            continue
        observed = firewall.get(key)
        if not _sha256_text(observed):
            blockers.append(f"evaluation_firewall.{key}_invalid")
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
    if firewall.get("decontaminated_record_inventory_digest_sha256") != corpus.get(
        "record_inventory_digest_sha256"
    ):
        blockers.append(
            "evaluation_firewall.decontaminated_record_inventory_digest_sha256_mismatch"
        )

    return sorted(set(blockers))
