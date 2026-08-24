from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.integration.dependency_evidence import (
    SupplyChainEvidenceError,
    build_supply_chain_documents,
    validate_supply_chain_evidence,
    write_supply_chain_documents,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "a" * 40


def _documents() -> tuple[dict, dict]:
    return build_supply_chain_documents(
        root=ROOT,
        profile_id="linux-x86_64",
        source_sha=SOURCE_SHA,
    )


def test_committed_lock_emits_deterministic_cyclonedx_with_explicit_unknowns() -> None:
    first_sbom, first_evidence = _documents()
    second_sbom, second_evidence = _documents()
    assert first_sbom == second_sbom
    assert first_evidence == second_evidence
    assert first_sbom["bomFormat"] == "CycloneDX"
    assert first_sbom["specVersion"] == "1.6"
    assert first_evidence["component_count"] == len(first_sbom["components"])
    assert first_evidence["component_count"] > 0
    assert first_evidence["vulnerability"]["status"] == "UNKNOWN"
    assert first_evidence["license"]["status"] == "UNKNOWN"


def test_unknown_evidence_validates_for_ci_but_fails_release_preflight(tmp_path: Path) -> None:
    sbom, evidence = _documents()
    sbom_path = tmp_path / "sbom.json"
    evidence_path = tmp_path / "evidence.json"
    write_supply_chain_documents(
        sbom=sbom,
        evidence=evidence,
        sbom_path=sbom_path,
        evidence_path=evidence_path,
    )
    validated = validate_supply_chain_evidence(
        root=ROOT,
        sbom_path=sbom_path,
        evidence_path=evidence_path,
        expected_source_sha=SOURCE_SHA,
        require_resolved=False,
    )
    assert validated["vulnerability"]["status"] == "UNKNOWN"
    with pytest.raises(SupplyChainEvidenceError, match="vulnerability=PASS"):
        validate_supply_chain_evidence(
            root=ROOT,
            sbom_path=sbom_path,
            evidence_path=evidence_path,
            expected_source_sha=SOURCE_SHA,
            require_resolved=True,
        )


def test_tampered_evidence_self_hash_is_rejected(tmp_path: Path) -> None:
    sbom, evidence = _documents()
    sbom_path = tmp_path / "sbom.json"
    evidence_path = tmp_path / "evidence.json"
    write_supply_chain_documents(
        sbom=sbom,
        evidence=evidence,
        sbom_path=sbom_path,
        evidence_path=evidence_path,
    )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    document["component_count"] += 1
    evidence_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SupplyChainEvidenceError, match="self-hash mismatch"):
        validate_supply_chain_evidence(
            root=ROOT,
            sbom_path=sbom_path,
            evidence_path=evidence_path,
            expected_source_sha=SOURCE_SHA,
            require_resolved=False,
        )


def test_source_sha_binding_is_enforced(tmp_path: Path) -> None:
    sbom, evidence = _documents()
    sbom_path = tmp_path / "sbom.json"
    evidence_path = tmp_path / "evidence.json"
    write_supply_chain_documents(
        sbom=sbom,
        evidence=evidence,
        sbom_path=sbom_path,
        evidence_path=evidence_path,
    )
    with pytest.raises(SupplyChainEvidenceError, match="source SHA mismatch"):
        validate_supply_chain_evidence(
            root=ROOT,
            sbom_path=sbom_path,
            evidence_path=evidence_path,
            expected_source_sha="b" * 40,
            require_resolved=False,
        )
